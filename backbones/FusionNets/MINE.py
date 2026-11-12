"""
MINE multimodal intent backbone (v1.3 + LOO relation importance).

Pipeline
--------
  text/video/audio → align → proj → temporal gate → pool
  → [optional] shared/private split
  → unimodal logits + EMC reliability r_m (and uncertainty u_m)
  → [optional] RelationImportanceBranch: Ĉ_m, g_m  (LOO-supervised)
  → fusion (weighted / tensor / set / cdpr) → joint logits
  → logits_fusion = 0.5·joint + 0.5·Σ signed_evidence   (v1.3 default)

Relation branch (use_relation_branch=True, needs shared/private):
  Z_m = MLP([H^s; H^p]);  Ĉ ≈ C_rel = U(all)−U(leave-m);
  g_m = Gate(r_m, Ĉ_m).  See relation_branch.py.
"""

import torch
from torch import nn
import torch.nn.functional as F

from ..SubNets.AlignNets import AlignSubNet
from ..SubNets.FeatureNets import BERTEncoder
from .temporal_gates import ContextTemporalGate
from .relation_branch import RelationImportanceBranch
from .advanced_fusion import build_fusion_module


class MINE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.orig_d_t = args.text_feat_dim
        self.orig_d_a = args.audio_feat_dim
        self.orig_d_v = args.video_feat_dim
        self.d_t = self.d_a = self.d_v = dim = args.dst_feature_dims
        self.output_dim = args.num_labels

        self.text_dropout = args.text_dropout
        self.output_dropout = args.output_dropout
        self.reliability_mode = getattr(args, 'reliability_mode', 'sigmoid')
        self.compute_rel_contrib = getattr(args, 'compute_rel_contrib', False)
        # v1.3 combines joint features with signed unimodal evidence
        self.use_evidence_fusion = bool(getattr(args, 'use_evidence_fusion', True))

        # ---- encoders / projectors ----
        self.text_subnet = BERTEncoder.from_pretrained(
            args.bert_path + args.text_backbone, cache_dir=args.cache_path
        )
        if args.need_aligned:
            self.alignNet = AlignSubNet(args, args.aligned_method)

        self.proj_t = nn.Conv1d(self.orig_d_t, self.d_t, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)

        # Keep v1.3 construction order (pred → temporal gates). Xavier init is
        # name-sorted, so this does not change weights; AlignNet LSTMs sit before
        # both and are unaffected either way.
        self.pred_t = nn.Linear(self.d_t, self.output_dim)
        self.pred_v = nn.Linear(self.d_v, self.output_dim)
        self.pred_a = nn.Linear(self.d_a, self.output_dim)
        self.pred_fusion = nn.Linear(dim, self.output_dim)

        self.la_t = ContextTemporalGate(self.d_t)
        self.la_v = ContextTemporalGate(self.d_v)
        self.la_a = ContextTemporalGate(self.d_a)

        # ---- shared / private (v1.3) ----
        self.use_shared_private = getattr(args, 'use_shared_private', False)
        if self.use_shared_private:
            sp_dropout = getattr(args, 'sp_dropout', self.output_dropout)
            self.shared = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(sp_dropout),
            )
            self.privatized_t = nn.Sequential(
                nn.Linear(self.d_t, self.d_t), nn.ReLU(), nn.Dropout(sp_dropout),
            )
            self.privatized_v = nn.Sequential(
                nn.Linear(self.d_v, self.d_v), nn.ReLU(), nn.Dropout(sp_dropout),
            )
            self.privatized_a = nn.Sequential(
                nn.Linear(self.d_a, self.d_a), nn.ReLU(), nn.Dropout(sp_dropout),
            )

        # ---- optional fusion backend (default: weighted sum) ----
        self.fusion_mode = str(getattr(args, 'fusion_mode', 'weighted')).lower()
        if self.fusion_mode == 'weight':
            self.fusion_mode = 'weighted'
        if self.fusion_mode == 'cdpr' and not self.use_shared_private:
            raise ValueError("fusion_mode='cdpr' requires use_shared_private=True")
        self.fusion_ext = build_fusion_module(
            self.fusion_mode, dim, args, default_dropout=self.output_dropout
        )
        # convenience aliases used by some scripts
        if self.fusion_mode == 'tensor':
            self.tensor_fusion = self.fusion_ext
        elif self.fusion_mode == 'set':
            self.set_fusion = self.fusion_ext
        elif self.fusion_mode == 'cdpr':
            self.cdpr_fusion = self.fusion_ext

        # ---- LOO relation importance (extension over v1.3) ----
        self.use_relation_branch = getattr(args, 'use_relation_branch', False)
        self.relation = None
        if self.use_relation_branch:
            if not self.use_shared_private:
                raise ValueError('use_relation_branch=True requires use_shared_private=True')
            self.relation = RelationImportanceBranch(
                dim, args, default_dropout=self.output_dropout
            )
            # Aliases so manager / relation_report keep working unchanged
            self.rel_beta = self.relation.rel_beta
            self.rel_alpha = self.relation.rel_alpha
            self.rel_phi = self.relation.rel_phi
            self.rel_fusion_norm = self.relation.rel_fusion_norm
            self.relation_content_mode = self.relation.content_mode
            self.relation_loo_use_gate = self.relation.loo_use_gate
            self.rel_z_residual = self.relation.z_residual

        self.init_weights()
        if self.relation is not None:
            self.relation.configure_after_init(
                alpha_init=getattr(args, 'relation_alpha_init', 1.0),
                beta_init=getattr(args, 'relation_beta_init', 0.0),
                warmup_epochs=getattr(args, 'relation_beta_warmup_epochs', 0),
            )

    # ------------------------------------------------------------------ init
    def init_weights(self):
        seed = getattr(self.args, 'seed', None)
        if seed is not None:
            torch.manual_seed(int(seed))
        skip = ('text_subnet',)
        modules = [
            (name, m) for name, m in self.named_modules()
            if name and not name.startswith(skip)
            and isinstance(m, (nn.Linear, nn.Conv1d))
        ]
        for _, m in sorted(modules, key=lambda x: x[0]):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0.)

    # ----------------------------------------------------------- public API
    def _compute_relation_gates(self, r_t, r_v, r_a, C_hat_t, C_hat_v, C_hat_a, beta=None):
        """Backward-compatible wrapper for relation_report offline sweeps. """
        return self.relation.compute_gates(
            r_t, r_v, r_a, C_hat_t, C_hat_v, C_hat_a, beta=beta
        )

    def forward(self, text_data, video_data, audio_data, label_ids=None):
        # 1) encode + temporal gate + pool
        h_t, h_v, h_a, gate_t, gate_v, gate_a = self._encode_temporal(
            text_data, video_data, audio_data
        )
        h_t_p, h_v_p, h_a_p = h_t.mean(0), h_v.mean(0), h_a.mean(0)

        # 2) shared / private (v1.3); unimodal heads use private when enabled
        shared_t = shared_v = shared_a = None
        private_t = private_v = private_a = None
        feat_t, feat_v, feat_a = h_t_p, h_v_p, h_a_p
        if self.use_shared_private:
            shared_t, shared_v, shared_a = (
                self.shared(h_t_p), self.shared(h_v_p), self.shared(h_a_p)
            )
            private_t = self.privatized_t(h_t_p)
            private_v = self.privatized_v(h_v_p)
            private_a = self.privatized_a(h_a_p)
            feat_t, feat_v, feat_a = private_t, private_v, private_a

        # 3) unimodal logits → disagreement → reliability / uncertainty
        logits_t = self.pred_t(feat_t)
        logits_v = self.pred_v(feat_v)
        logits_a = self.pred_a(feat_a)
        d_t, d_v, d_a = self._sample_disagreement(logits_t, logits_v, logits_a)
        E_t, E_v, E_a = self.compute_emc(
            h_t, h_v, h_a,
            logits_t.unsqueeze(0), logits_v.unsqueeze(0), logits_a.unsqueeze(0),
            d_t.unsqueeze(0), d_v.unsqueeze(0), d_a.unsqueeze(0),
        )
        r_t, r_v, r_a = self._reliability_from_energy(E_t, E_v, E_a)
        u_t, u_v, u_a = torch.sigmoid(E_t), torch.sigmoid(E_v), torch.sigmoid(E_a)
        e_t = self._signed_evidence(logits_t)
        e_v = self._signed_evidence(logits_v)
        e_a = self._signed_evidence(logits_a)

        # 4) relation branch → Ĉ, g  (optional)
        rel_out = None
        g_t = g_v = g_a = None
        Z_t = Z_v = Z_a = None
        C_hat_t = C_hat_v = C_hat_a = None
        if self.relation is not None:
            rel_out = self.relation(
                shared_t, shared_v, shared_a,
                private_t, private_v, private_a,
                r_t, r_v, r_a,
            )
            Z_t, Z_v, Z_a = rel_out['Z_t'], rel_out['Z_v'], rel_out['Z_a']
            C_hat_t, C_hat_v, C_hat_a = (
                rel_out['C_hat_t'], rel_out['C_hat_v'], rel_out['C_hat_a']
            )
            g_t, g_v, g_a = rel_out['g_t'], rel_out['g_v'], rel_out['g_a']

        # 5) multimodal fusion
        feat_fusion, extra_fuse = self._fuse_modalities(
            h_t_p, h_v_p, h_a_p,
            shared_t, shared_v, shared_a,
            private_t, private_v, private_a,
            feat_t, feat_v, feat_a,
            r_t, r_v, r_a, g_t, g_v, g_a,
            u_t, u_v, u_a, d_t, d_v, d_a,
            Z_t, Z_v, Z_a,
        )

        logits_joint = self.pred_fusion(feat_fusion)
        if self.use_evidence_fusion:
            logits_fusion = 0.5 * logits_joint + 0.5 * (e_t + e_v + e_a)
        else:
            logits_fusion = logits_joint

        res = {
            'feat_t': feat_t, 'feat_v': feat_v, 'feat_a': feat_a,
            'feat_fusion': feat_fusion,
            'logits_t': logits_t, 'logits_v': logits_v, 'logits_a': logits_a,
            'logits_fusion': logits_fusion,
            'gate_t': gate_t, 'gate_v': gate_v, 'gate_a': gate_a,
            'r_t': r_t, 'r_v': r_v, 'r_a': r_a,
            'u_t': u_t, 'u_v': u_v, 'u_a': u_a,
            'energy_t': E_t, 'energy_v': E_v, 'energy_a': E_a,
            'fusion_mode': self.fusion_mode,
        }
        res.update(extra_fuse)
        if self.use_shared_private:
            res.update({
                'shared_t': shared_t, 'shared_v': shared_v, 'shared_a': shared_a,
                'private_t': private_t, 'private_v': private_v, 'private_a': private_a,
            })
        if rel_out is not None:
            res.update({
                **{k: rel_out[k] for k in (
                    'Z_t', 'Z_v', 'Z_a',
                    'C_hat_t', 'C_hat_v', 'C_hat_a',
                    'g_t', 'g_v', 'g_a',
                )},
                'I_t': r_t, 'I_v': r_v, 'I_a': r_a,
            })

        # 6) LOO C_rel supervision / pre-analysis
        need_loo = label_ids is not None and (
            self.compute_rel_contrib or self.relation is not None
        )
        if need_loo:
            res.update(self._attach_loo_targets(
                label_ids, feat_t, feat_v, feat_a,
                r_t, r_v, r_a, g_t, g_v, g_a,
                e_t, e_v, e_a, Z_t, Z_v, Z_a,
            ))
        return res

    # ---------------------------------------------------------- encode
    def _encode_temporal(self, text_data, video_data, audio_data):
        text_feats = self.text_subnet(text_data)
        if self.args.need_aligned:
            x_t, x_v, x_a = self.alignNet(text_feats, video_data, audio_data)
        else:
            x_t, x_v, x_a = text_feats, video_data, audio_data

        x_t = F.dropout(x_t.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_v = x_v.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        proj_t = x_t if self.orig_d_t == self.d_t else self.proj_t(x_t)
        proj_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_t, proj_v, proj_a = (
            proj_t.permute(2, 0, 1),
            proj_v.permute(2, 0, 1),
            proj_a.permute(2, 0, 1),
        )
        h_t, gate_t = self.la_t(proj_t)
        h_v, gate_v = self.la_v(proj_v)
        h_a, gate_a = self.la_a(proj_a)
        return h_t, h_v, h_a, gate_t, gate_v, gate_a

    # ---------------------------------------------------------- reliability
    def _reliability_from_energy(self, E_t, E_v, E_a):
        """r ≥ 0.  sigmoid: σ(-E); exp: e^{-E}; exp_softmax: softmax over mods."""
        if self.reliability_mode == 'sigmoid':
            return torch.sigmoid(-E_t), torch.sigmoid(-E_v), torch.sigmoid(-E_a)
        r_t, r_v, r_a = torch.exp(-E_t), torch.exp(-E_v), torch.exp(-E_a)
        if self.reliability_mode == 'exp_softmax':
            stack = F.softmax(torch.stack([r_t, r_v, r_a], dim=-1), dim=-1)
            r_t, r_v, r_a = stack.unbind(dim=-1)
        return r_t, r_v, r_a

    @staticmethod
    def _signed_evidence(logits):
        C = logits.size(-1)
        if C <= 1:
            raise ValueError('num_labels must be > 1')
        return C / (C - 1) * (logits - logits.mean(dim=-1, keepdim=True))

    def _sample_disagreement(self, logits_t, logits_v, logits_a):
        d_t, d_v, d_a = self._cross_modal_disagreement(
            logits_t.unsqueeze(0), logits_v.unsqueeze(0), logits_a.unsqueeze(0),
        )
        return d_t.squeeze(0), d_v.squeeze(0), d_a.squeeze(0)

    # ---------------------------------------------------------- fusion
    def _fuse_modalities(
        self,
        h_t_p, h_v_p, h_a_p,
        shared_t, shared_v, shared_a,
        private_t, private_v, private_a,
        feat_t, feat_v, feat_a,
        r_t, r_v, r_a, g_t, g_v, g_a,
        u_t, u_v, u_a, d_t, d_v, d_a,
        Z_t, Z_v, Z_a,
    ):
        """Return (feat_fusion [N,D], extra_dict)."""
        extra = {}
        # Importance for backends that accept a weight: prefer g when relation is on
        # for weighted/cdpr; tensor/set need fusion_use_relation_gate=True.
        w_t, w_v, w_a = r_t, r_v, r_a
        use_g = (
            g_t is not None and (
                self.fusion_mode in ('weighted', 'cdpr')
                or bool(getattr(self.args, 'fusion_use_relation_gate', False))
            )
        )
        if use_g:
            w_t, w_v, w_a = g_t, g_v, g_a

        if self.fusion_mode == 'tensor':
            return self.fusion_ext(
                feat_t, feat_v, feat_a, w_t, w_v, w_a, u_t, u_v, u_a
            ), extra

        if self.fusion_mode == 'set':
            return self.fusion_ext(
                feat_t, feat_v, feat_a, w_t, w_v, w_a, u_t, u_v, u_a
            ), extra

        if self.fusion_mode == 'cdpr':
            fused, lam = self.fusion_ext(
                h_t_p, h_v_p, h_a_p,
                shared_t, shared_v, shared_a,
                private_t, private_v, private_a,
                w_t, w_v, w_a, u_t, u_v, u_a, d_t, d_v, d_a,
            )
            extra['cdpr_lambda'] = lam
            return fused, extra

        # --- weighted (v1.3 + optional relation gate) ---
        if self.relation is not None and g_t is not None:
            c_t, c_v, c_a = self.relation.fusion_content(
                feat_t, feat_v, feat_a, Z_t, Z_v, Z_a
            )
            return self.relation.weighted_fuse(g_t, g_v, g_a, c_t, c_v, c_a), extra

        fused = (
            r_t.unsqueeze(-1) * feat_t
            + r_v.unsqueeze(-1) * feat_v
            + r_a.unsqueeze(-1) * feat_a
        )
        return fused, extra

    def _attach_loo_targets(
        self, label_ids,
        feat_t, feat_v, feat_a,
        r_t, r_v, r_a, g_t, g_v, g_a,
        e_t, e_v, e_a,
        Z_t, Z_v, Z_a,
    ):
        """Compute detached C_rel via leave-one-modality-out margins."""
        # Content for LOO must match what the main fusion mixes.
        if (
            self.relation is not None
            and g_t is not None
            and self.relation.content_mode in ('z', 'mix')
        ):
            feats = self.relation.fusion_content(
                feat_t, feat_v, feat_a, Z_t, Z_v, Z_a
            )
        else:
            feats = (feat_t, feat_v, feat_a)

        if (
            self.relation is not None
            and self.relation.loo_use_gate
            and g_t is not None
        ):
            rs = (g_t.detach(), g_v.detach(), g_a.detach())
        else:
            rs = (r_t, r_v, r_a)

        if self.relation is not None:
            loo = self.relation.compute_loo_C_rel(
                feats, rs, (e_t, e_v, e_a), label_ids, self._fuse_subset
            )
        else:
            # pre_analysis without a trainable relation module
            loo = self._loo_fallback(feats, rs, (e_t, e_v, e_a), label_ids)
        loo.update({'I_t': r_t, 'I_v': r_v, 'I_a': r_a})
        return loo

    def _loo_fallback(self, feats, rs, es, label_ids):
        """LOO when relation module is off but compute_rel_contrib=True."""
        logits_tav, _ = self._fuse_subset(feats, rs, es, (True, True, True))
        logits_av, _ = self._fuse_subset(feats, rs, es, (False, True, True))
        logits_tv, _ = self._fuse_subset(feats, rs, es, (True, True, False))
        logits_ta, _ = self._fuse_subset(feats, rs, es, (True, False, True))
        U_tav = RelationImportanceBranch.classification_margin(logits_tav, label_ids)
        U_av = RelationImportanceBranch.classification_margin(logits_av, label_ids)
        U_tv = RelationImportanceBranch.classification_margin(logits_tv, label_ids)
        U_ta = RelationImportanceBranch.classification_margin(logits_ta, label_ids)
        return {
            'C_rel_t': (U_tav - U_av).detach(),
            'C_rel_v': (U_tav - U_ta).detach(),
            'C_rel_a': (U_tav - U_tv).detach(),
            'U_tav': U_tav.detach(), 'U_av': U_av.detach(),
            'U_tv': U_tv.detach(), 'U_ta': U_ta.detach(),
        }

    def _renorm_reliability(self, rs, mask):
        stack = torch.stack(list(rs), dim=-1)
        m = stack.new_tensor([float(x) for x in mask])
        masked = stack * m.view(1, -1)
        if self.reliability_mode == 'exp_softmax':
            masked = masked.masked_fill(m.view(1, -1) == 0, float('-inf'))
            return F.softmax(masked, dim=-1).unbind(dim=-1)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (masked / denom).unbind(dim=-1)

    def _fuse_subset(self, feats, rs, es, mask):
        """
        Subset fusion for LOO. Matches main weighted path (+ evidence if enabled).
        Full-set keeps original r; proper subsets re-normalize over active mods.
        """
        if all(mask):
            r_t, r_v, r_a = rs
        else:
            r_t, r_v, r_a = self._renorm_reliability(rs, mask)
        feat_t, feat_v, feat_a = feats
        e_t, e_v, e_a = es
        use_t, use_v, use_a = mask

        feat = feat_t.new_zeros(feat_t.shape)
        evid = e_t.new_zeros(e_t.shape)
        if use_t:
            feat = feat + r_t.unsqueeze(-1) * feat_t
            evid = evid + e_t
        if use_v:
            feat = feat + r_v.unsqueeze(-1) * feat_v
            evid = evid + e_v
        if use_a:
            feat = feat + r_a.unsqueeze(-1) * feat_a
            evid = evid + e_a

        logits_joint = self.pred_fusion(feat)
        if self.use_evidence_fusion:
            logits = 0.5 * logits_joint + 0.5 * evid
        else:
            logits = logits_joint
        return logits, feat

    # ---------------------------------------------------------- EMC / JSD
    def _js_divergence(self, logits_p, logits_q):
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        p, q = log_p.exp(), log_q.exp()
        log_mix = torch.logaddexp(log_p, log_q) - torch.log(logits_p.new_tensor(2.0))
        js = 0.5 * (p * (log_p - log_mix)).sum(-1) + 0.5 * (q * (log_q - log_mix)).sum(-1)
        return js.clamp_min(0.0)

    def _cross_modal_disagreement(self, logits_t, logits_v, logits_a):
        d_t = 0.5 * (self._js_divergence(logits_t, logits_v) + self._js_divergence(logits_t, logits_a))
        d_v = 0.5 * (self._js_divergence(logits_v, logits_t) + self._js_divergence(logits_v, logits_a))
        d_a = 0.5 * (self._js_divergence(logits_a, logits_t) + self._js_divergence(logits_a, logits_v))
        return d_t, d_v, d_a

    def compute_emc(self, h_t, h_v, h_a, logits_t, logits_v, logits_a, d_t, d_v, d_a):
        """
        Sample-level energy from predictive entropy + cross-modal disagreement.
        Higher energy ⇒ lower reliability. Returns (E_t, E_v, E_a) each [N].
        """
        beta = getattr(self.args, 'emc_beta', 1.0)
        gamma = getattr(self.args, 'emc_gamma', 1.0)
        energies = []
        for h, logits, d_m in (
            (h_t, logits_t, d_t), (h_v, logits_v, d_v), (h_a, logits_a, d_a),
        ):
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            entropy = -(p * logp).sum(dim=-1).mean(dim=0)  # [N]
            logC = logits.new_tensor(logits.size(-1)).log()
            u_pred = entropy / logC
            d_scalar = d_m.mean(dim=0)
            energies.append(beta * u_pred + gamma * d_scalar)
        return energies
