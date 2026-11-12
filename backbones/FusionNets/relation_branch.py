"""
LOO relational importance branch (design: 261112关系重要性).

Descriptor (default, no heavy Z MLP):
  q_m = W_s s_m + W_p p_m
  R̂_m = f_rel(q_m, Agg_{j≠m} ψ(q_m, q_j))

Legacy: relation_desc_mode='mlp' keeps Z_m = MLP([s;p]).

Gates (relation_gate_mode):
  multiplicative | additive | softmax_mix | signed_exp

Fusion content when relation is on (relation_fusion_mode):
  weighted   — g ⊙ content (private / q / mix), v1.3-compatible
  sp_concat  — F = MLP([Agg(s); Σ_m w_m p'_m])   ← design §2.2

LOO target C_rel_m = U(all)−U(leave-m) is stop-grad; prefer EMA teacher
in the manager so early head noise does not dominate.
"""

import torch
from torch import nn
import torch.nn.functional as F


class RelationImportanceBranch(nn.Module):
    """Predict signed modality contribution R̂ and build fusion weights w."""

    def __init__(self, dim, args, default_dropout=0.2):
        super().__init__()
        dropout = getattr(args, 'relation_dropout', default_dropout)
        self.dim = dim

        desc_mode = str(getattr(args, 'relation_desc_mode', 'linear')).lower()
        if desc_mode not in ('linear', 'mlp'):
            raise ValueError("relation_desc_mode must be linear|mlp")
        self.desc_mode = desc_mode

        if desc_mode == 'linear':
            # q_m = W_s s_m + W_p p_m  (shared across modalities)
            self.W_s = nn.Linear(dim, dim, bias=False)
            self.W_p = nn.Linear(dim, dim, bias=False)
            self.rel_in = None
        else:
            self.W_s = self.W_p = None
            self.rel_in = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.rel_psi = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.rel_score = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        # Optional map q → fusion content (content_mode z|mix)
        self.rel_phi = nn.Linear(dim, dim)

        content_mode = str(getattr(args, 'relation_content_mode', 'private')).lower()
        if content_mode not in ('private', 'z', 'mix'):
            raise ValueError(
                f"relation_content_mode must be private|z|mix, got {content_mode}"
            )
        self.content_mode = content_mode
        self.z_residual = float(getattr(args, 'relation_z_residual', 0.0))
        self.loo_use_gate = bool(getattr(args, 'relation_loo_use_gate', False))

        gate_mode = str(getattr(args, 'relation_gate_mode', 'signed_exp')).lower()
        if gate_mode not in ('multiplicative', 'additive', 'softmax_mix', 'signed_exp'):
            raise ValueError(
                "relation_gate_mode must be "
                "multiplicative|additive|softmax_mix|signed_exp"
            )
        self.gate_mode = gate_mode
        self.conf_gate = bool(getattr(args, 'relation_conf_gate', True))
        self.conf_scale = float(getattr(args, 'relation_conf_scale', 2.0))
        self.conf_bias = float(getattr(args, 'relation_conf_bias', 0.5))
        self.signed_exp_clip = float(getattr(args, 'relation_signed_exp_clip', 2.0))
        self.gate_renorm = bool(getattr(args, 'relation_gate_renorm', True))

        fusion_mode = str(getattr(args, 'relation_fusion_mode', 'sp_concat')).lower()
        if fusion_mode not in ('weighted', 'sp_concat'):
            raise ValueError("relation_fusion_mode must be weighted|sp_concat")
        self.fusion_mode = fusion_mode
        if fusion_mode == 'sp_concat':
            self.fuse_private = nn.Linear(dim, dim)
            self.fuse_out = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.fuse_private = None
            self.fuse_out = None

        use_norm = bool(getattr(args, 'relation_use_fusion_norm', False))
        self.rel_fusion_norm = nn.LayerNorm(dim) if use_norm else nn.Identity()

        alpha_init = float(getattr(args, 'relation_alpha_init', 1.0))
        beta_init = float(getattr(args, 'relation_beta_init', 0.0))
        self.alpha_fixed = bool(getattr(args, 'relation_alpha_fixed', True))
        self.beta_fixed = bool(getattr(args, 'relation_beta_fixed', True))
        self.rel_alpha = nn.Parameter(
            torch.tensor(alpha_init), requires_grad=not self.alpha_fixed
        )
        self.rel_beta = nn.Parameter(
            torch.tensor(beta_init), requires_grad=not self.beta_fixed
        )
        self._warmup_epochs = int(getattr(args, 'relation_beta_warmup_epochs', 0))
        self._beta_target = float(getattr(args, 'relation_beta_init', 0.0))

    def configure_after_init(self, alpha_init=None, beta_init=None, warmup_epochs=None):
        """Call after parent xavier init: set φ / α / β to intended starts."""
        if self.content_mode == 'z':
            nn.init.eye_(self.rel_phi.weight)
            nn.init.zeros_(self.rel_phi.bias)
        else:
            nn.init.zeros_(self.rel_phi.weight)
            nn.init.zeros_(self.rel_phi.bias)

        if self.desc_mode == 'linear':
            # Start near q ≈ s + p (identity mix); private/shared already separated.
            nn.init.eye_(self.W_s.weight)
            nn.init.eye_(self.W_p.weight)

        if self.fusion_mode == 'sp_concat':
            nn.init.eye_(self.fuse_private.weight)
            nn.init.zeros_(self.fuse_private.bias)

        alpha = float(self.rel_alpha.detach()) if alpha_init is None else float(alpha_init)
        beta_tgt = self._beta_target if beta_init is None else float(beta_init)
        warm = self._warmup_epochs if warmup_epochs is None else int(warmup_epochs)
        with torch.no_grad():
            self.rel_alpha.fill_(alpha)
            self.rel_beta.fill_(0.0 if warm > 0 else beta_tgt)
        self.rel_alpha.requires_grad = not self.alpha_fixed
        self.rel_beta.requires_grad = not self.beta_fixed

    def build_Z(self, shared_t, shared_v, shared_a, private_t, private_v, private_a):
        """Build modality descriptors q_m (also exposed as Z_* for compat)."""
        if self.desc_mode == 'linear':
            q_t = self.W_s(shared_t) + self.W_p(private_t)
            q_v = self.W_s(shared_v) + self.W_p(private_v)
            q_a = self.W_s(shared_a) + self.W_p(private_a)
            return q_t, q_v, q_a
        Z_t = self.rel_in(torch.cat([shared_t, private_t], dim=-1))
        Z_v = self.rel_in(torch.cat([shared_v, private_v], dim=-1))
        Z_a = self.rel_in(torch.cat([shared_a, private_a], dim=-1))
        return Z_t, Z_v, Z_a

    def _psi(self, Z_m, Z_n):
        return self.rel_psi(torch.cat(
            [Z_m, Z_n, (Z_m - Z_n).abs(), Z_m * Z_n], dim=-1
        ))

    def predict_C_hat(self, Z_t, Z_v, Z_a):
        """R̂_m = f_r([q_m; mean_{n≠m} Q_mn]), signed / unbounded."""
        Q_tv = self._psi(Z_t, Z_v)
        Q_ta = self._psi(Z_t, Z_a)
        Q_vt = self._psi(Z_v, Z_t)
        Q_va = self._psi(Z_v, Z_a)
        Q_at = self._psi(Z_a, Z_t)
        Q_av = self._psi(Z_a, Z_v)

        Q_t = 0.5 * (Q_tv + Q_ta)
        Q_v = 0.5 * (Q_vt + Q_va)
        Q_a = 0.5 * (Q_at + Q_av)

        C_hat_t = self.rel_score(torch.cat([Z_t, Q_t], dim=-1)).squeeze(-1)
        C_hat_v = self.rel_score(torch.cat([Z_v, Q_v], dim=-1)).squeeze(-1)
        C_hat_a = self.rel_score(torch.cat([Z_a, Q_a], dim=-1)).squeeze(-1)
        return C_hat_t, C_hat_v, C_hat_a

    @staticmethod
    def modality_standardize(x):
        """Per-sample z-score over the modality axis. x: [N, 3]."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    def compute_gates(self, r_t, r_v, r_a, C_hat_t, C_hat_v, C_hat_a, beta=None):
        """
        Combine intrinsic reliability I≡r with predicted LOO scores R̂.

        multiplicative: g = r · (1 + β · tanh(R̃) · conf)
        additive:       g = σ(α · logit(r) + β · R̃ · conf)
        softmax_mix:    g = softmax(α·r + β·R̃)             # relative, ≥0
        signed_exp:     g = σ(α·r) · exp(clip(β·R̃, ±c))   # keeps negative R̂ as suppress
        """
        r_stack = torch.stack([r_t, r_v, r_a], dim=-1).clamp_min(1e-4)
        C_stack = torch.stack([C_hat_t, C_hat_v, C_hat_a], dim=-1)
        C_norm = self.modality_standardize(C_stack)

        beta_t = self.rel_beta if beta is None else beta
        if not torch.is_tensor(beta_t):
            beta_t = r_stack.new_tensor(float(beta_t))

        if self.conf_gate and self.gate_mode in ('multiplicative', 'additive'):
            conf = torch.sigmoid(self.conf_scale * (C_norm.abs() - self.conf_bias))
        else:
            conf = C_norm.new_ones(C_norm.shape)

        if self.gate_mode == 'softmax_mix':
            logits = self.rel_alpha * r_stack + beta_t * C_norm
            g_stack = F.softmax(logits, dim=-1)
        elif self.gate_mode == 'signed_exp':
            # σ(α I) · exp(clip(β R̃)); optional sum-renorm to keep scale ~1
            amp = torch.sigmoid(self.rel_alpha * r_stack)
            mod = (beta_t * C_norm).clamp(
                -self.signed_exp_clip, self.signed_exp_clip
            )
            g_stack = amp * torch.exp(mod)
            if self.gate_renorm:
                g_stack = g_stack / g_stack.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                g_stack = g_stack * g_stack.size(-1)  # mean ≈ 1 like softmax*L
        elif self.gate_mode == 'additive':
            mod = beta_t * C_norm * conf
            r_prob = (r_stack / (1.0 + r_stack)).clamp(1e-4, 1.0 - 1e-4)
            I_logit = torch.log(r_prob) - torch.log1p(-r_prob)
            g_stack = torch.sigmoid(self.rel_alpha * I_logit + mod)
        else:
            mod = beta_t * torch.tanh(C_norm) * conf
            scale = (1.0 + mod).clamp(min=1e-4)
            g_stack = (r_stack * scale).clamp_min(1e-4)
        return g_stack.unbind(dim=-1)

    def fusion_content(self, feat_t, feat_v, feat_a, Z_t, Z_v, Z_a):
        """Content that g_m multiplies in the weighted fusion path."""
        if self.content_mode == 'z':
            return self.rel_phi(Z_t), self.rel_phi(Z_v), self.rel_phi(Z_a)
        if self.content_mode == 'mix':
            lam = self.z_residual
            return (
                feat_t + lam * self.rel_phi(Z_t),
                feat_v + lam * self.rel_phi(Z_v),
                feat_a + lam * self.rel_phi(Z_a),
            )
        return feat_t, feat_v, feat_a

    def weighted_fuse(self, g_t, g_v, g_a, c_t, c_v, c_a):
        fused = (
            g_t.unsqueeze(-1) * c_t
            + g_v.unsqueeze(-1) * c_v
            + g_a.unsqueeze(-1) * c_a
        )
        return self.rel_fusion_norm(fused)

    def sp_fuse(self, shareds, privates, weights, mask=(True, True, True)):
        """
        Design §2.2: F = MLP([Agg(s); Σ_m w_m p'_m]) over active modalities.
        shareds/privates/weights: length-3 sequences; mask selects leave-in mods.
        """
        if self.fusion_mode != 'sp_concat':
            raise RuntimeError('sp_fuse requires relation_fusion_mode=sp_concat')
        S = None
        P = None
        n = 0
        for use, s, p, w in zip(mask, shareds, privates, weights):
            if not use:
                continue
            n += 1
            S = s if S is None else S + s
            term = w.unsqueeze(-1) * self.fuse_private(p)
            P = term if P is None else P + term
        if n == 0:
            raise ValueError('sp_fuse mask selects no modalities')
        S = S / float(n)
        fused = self.fuse_out(torch.cat([S, P], dim=-1))
        return self.rel_fusion_norm(fused)

    @staticmethod
    def classification_margin(logits, y):
        """U = z_y - max_{k≠y} z_k."""
        y = y.long()
        z_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        logits_masked = logits.clone()
        logits_masked.scatter_(1, y.unsqueeze(1), float('-inf'))
        z_other = logits_masked.max(dim=1).values
        return z_y - z_other

    def compute_loo_C_rel(self, feats, rs, es, label_ids, fuse_subset_fn):
        """
        C_rel_m = U(all) - U(leave-m-out), detached (supervision target only).

        fuse_subset_fn(feats, rs, es, mask) -> (logits, feat)
        mask: (use_t, use_v, use_a)
        """
        logits_tav, _ = fuse_subset_fn(feats, rs, es, (True, True, True))
        logits_av, _ = fuse_subset_fn(feats, rs, es, (False, True, True))
        logits_tv, _ = fuse_subset_fn(feats, rs, es, (True, True, False))
        logits_ta, _ = fuse_subset_fn(feats, rs, es, (True, False, True))

        U_tav = self.classification_margin(logits_tav, label_ids)
        U_av = self.classification_margin(logits_av, label_ids)
        U_tv = self.classification_margin(logits_tv, label_ids)
        U_ta = self.classification_margin(logits_ta, label_ids)

        return {
            'C_rel_t': (U_tav - U_av).detach(),
            'C_rel_v': (U_tav - U_ta).detach(),
            'C_rel_a': (U_tav - U_tv).detach(),
            'U_tav': U_tav.detach(),
            'U_av': U_av.detach(),
            'U_tv': U_tv.detach(),
            'U_ta': U_ta.detach(),
        }

    def forward(self, shared_t, shared_v, shared_a, private_t, private_v, private_a,
                r_t, r_v, r_a):
        """Returns dict with Z_*/q_*, C_hat_*, g_*."""
        Z_t, Z_v, Z_a = self.build_Z(
            shared_t, shared_v, shared_a, private_t, private_v, private_a
        )
        C_hat_t, C_hat_v, C_hat_a = self.predict_C_hat(Z_t, Z_v, Z_a)
        g_t, g_v, g_a = self.compute_gates(
            r_t, r_v, r_a, C_hat_t, C_hat_v, C_hat_a
        )
        return {
            'Z_t': Z_t, 'Z_v': Z_v, 'Z_a': Z_a,
            'q_t': Z_t, 'q_v': Z_v, 'q_a': Z_a,
            'C_hat_t': C_hat_t, 'C_hat_v': C_hat_v, 'C_hat_a': C_hat_a,
            'g_t': g_t, 'g_v': g_v, 'g_a': g_a,
        }
