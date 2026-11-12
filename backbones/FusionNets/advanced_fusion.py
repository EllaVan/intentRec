"""
Optional multimodal fusion backends (selected by args.fusion_mode).

Default MINE path uses simple reliability-weighted sum (see MINE.py).
These modules are extension points:

  - TensorBasedFusion ('tensor'): unary + pairwise (+triple) Hadamard terms
  - SetBasedFusion    ('set'):    Deep Sets / SAB over modality tokens
  - CDPRStyleFusion   ('cdpr'):   intuition(shared) + gated-private reasoning
"""

import torch
from torch import nn
import torch.nn.functional as F


def build_fusion_module(mode, dim, args, default_dropout=0.2):
    """Factory for optional fusion backends. Returns None for 'weighted'."""
    mode = str(mode).lower()
    if mode == 'weight':
        mode = 'weighted'
    dropout = getattr(args, 'fusion_dropout', default_dropout)
    if mode == 'weighted':
        return None
    if mode == 'tensor':
        return TensorBasedFusion(
            dim=dim,
            dropout=dropout,
            use_triple=bool(getattr(args, 'tensor_use_triple', True)),
            share_unary=bool(getattr(args, 'tensor_share_unary', False)),
        )
    if mode == 'set':
        return SetBasedFusion(
            dim=dim,
            dropout=dropout,
            set_type=getattr(args, 'set_fusion_type', 'sab'),
            num_heads=int(getattr(args, 'set_num_heads', getattr(args, 'nheads', 4))),
            num_layers=int(getattr(args, 'set_num_layers', 1)),
        )
    if mode == 'cdpr':
        return CDPRStyleFusion(dim=dim, dropout=dropout)
    raise ValueError(f"Unknown fusion_mode={mode!r}; use weighted|tensor|set|cdpr")


def _pair_ru(r_i, r_j, u_i, u_j):
    """Pair importance / uncertainty from unary r, u. All [N]."""
    r_ij = (r_i.clamp_min(1e-8) * r_j.clamp_min(1e-8)).sqrt()
    # high if either modality is uncertain
    u_ij = 1.0 - ((1.0 - u_i).clamp_min(0.0) * (1.0 - u_j).clamp_min(0.0)).sqrt()
    return r_ij, u_ij


def _gate(r, u):
    """Scalar gate r(1-u), shape [N, 1]."""
    return (r * (1.0 - u)).unsqueeze(-1)


class TensorBasedFusion(nn.Module):
    """
    Symmetric tensor-style fusion with low-rank Hadamard interactions
    (avoids full bilinear D×D×D cost).
    """

    def __init__(self, dim, dropout=0.1, use_triple=True, share_unary=False):
        super().__init__()
        self.dim = dim
        self.use_triple = bool(use_triple)

        def _unary():
            return nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if share_unary:
            self.unary = _unary()
            self.unary_t = self.unary_v = self.unary_a = self.unary
        else:
            self.unary_t = _unary()
            self.unary_v = _unary()
            self.unary_a = _unary()

        # H_ij = W(H_i ⊙ H_j)
        self.pair_ta = nn.Linear(dim, dim)
        self.pair_tv = nn.Linear(dim, dim)
        self.pair_av = nn.Linear(dim, dim)

        if self.use_triple:
            self.triple_tav = nn.Linear(dim, dim)

        self.out_norm = nn.LayerNorm(dim)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, h_t, h_v, h_a, r_t, r_v, r_a, u_t, u_v, u_a):
        """
        h_*: [N, D]; r_*, u_*: [N]
        """
        Ht = self.unary_t(h_t)
        Hv = self.unary_v(h_v)
        Ha = self.unary_a(h_a)

        fused = (
            _gate(r_t, u_t) * Ht
            + _gate(r_v, u_v) * Hv
            + _gate(r_a, u_a) * Ha
        )

        r_ta, u_ta = _pair_ru(r_t, r_a, u_t, u_a)
        r_tv, u_tv = _pair_ru(r_t, r_v, u_t, u_v)
        r_av, u_av = _pair_ru(r_a, r_v, u_a, u_v)

        fused = fused + _gate(r_ta, u_ta) * self.pair_ta(Ht * Ha)
        fused = fused + _gate(r_tv, u_tv) * self.pair_tv(Ht * Hv)
        fused = fused + _gate(r_av, u_av) * self.pair_av(Ha * Hv)

        if self.use_triple:
            r_tav = (r_t.clamp_min(1e-8) * r_v.clamp_min(1e-8) * r_a.clamp_min(1e-8)).pow(1.0 / 3.0)
            u_tav = 1.0 - (
                (1.0 - u_t).clamp_min(0.0)
                * (1.0 - u_v).clamp_min(0.0)
                * (1.0 - u_a).clamp_min(0.0)
            ).pow(1.0 / 3.0)
            fused = fused + _gate(r_tav, u_tav) * self.triple_tav(Ht * Hv * Ha)

        return self.out_drop(self.out_norm(fused))


class _SAB(nn.Module):
    """Single Self-Attention Block over a modality set (no positional encoding)."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [M, N, D]  (M modalities)
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class CDPRStyleFusion(nn.Module):
    """
    CDPR-style dual-path fusion (see A1referCodes/CDPR/.../model.py):

      z_intuition  = Norm( W_raw([H_t;H_v;H_a]) + α · Cons([Hs_t⊙Hs_v, ...]) )
      z_reasoning  = Σ_m w_m · H_m^p          # gates ONLY on private
      H_fusion     = (1-λ) z_intuition + λ z_reasoning

    λ is driven by mean modality uncertainty (and optional disagreement).
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.raw_fusion_layer = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Dropout(dropout),
        )
        self.consensus_extractor = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.consensus_scale = nn.Parameter(torch.zeros(1))
        self.intuition_norm = nn.LayerNorm(dim)
        # λ = σ(gate_sem + gate_stat + bias); start bias negative → lean intuition
        self.fusion_bias = nn.Parameter(torch.tensor(-1.0))
        self.unc_scale = nn.Parameter(torch.tensor(1.0))
        self.dis_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_t, h_v, h_a, shared_t, shared_v, shared_a,
                private_t, private_v, private_a,
                w_t, w_v, w_a, u_t=None, u_v=None, u_a=None,
                d_t=None, d_v=None, d_a=None):
        """
        h_*/shared_*/private_*: [N, D]
        w_*: [N] importance gates for private (r or g)
        u_*/d_*: [N] optional uncertainty / disagreement for λ
        """
        # --- intuition (CDPR L103-110): raw + shared pairwise consensus ---
        z_raw = self.raw_fusion_layer(torch.cat([h_t, h_v, h_a], dim=-1))
        sim_feat = torch.cat(
            [shared_t * shared_v, shared_t * shared_a, shared_v * shared_a], dim=-1
        )
        e_consensus = self.consensus_extractor(sim_feat)
        z_intuition = self.intuition_norm(z_raw + self.consensus_scale * e_consensus)

        # --- reasoning (CDPR L158): gates only on private ---
        z_reasoning = (
            w_t.unsqueeze(-1) * private_t
            + w_v.unsqueeze(-1) * private_v
            + w_a.unsqueeze(-1) * private_a
        )

        # --- mix (CDPR L167) ---
        if u_t is not None:
            u_mean = (u_t + u_v + u_a) / 3.0
        else:
            u_mean = w_t.new_zeros(w_t.shape)
        if d_t is not None:
            d_mean = (d_t + d_v + d_a) / 3.0
        else:
            d_mean = w_t.new_zeros(w_t.shape)
        gate_logits = (
            self.unc_scale * u_mean
            + self.dis_scale * d_mean
            + self.fusion_bias
        )
        lam = torch.sigmoid(gate_logits).unsqueeze(-1)  # [N, 1]
        return (1.0 - lam) * z_intuition + lam * z_reasoning, lam.squeeze(-1)


class SetBasedFusion(nn.Module):
    """
    Permutation-invariant set fusion.
    set_type:
      - 'deepset': ρ(Σ_m φ(z_m))
      - 'sab': shared self-attention over modalities + mean pool
    """

    def __init__(self, dim, dropout=0.1, set_type='sab', num_heads=4, num_layers=1):
        super().__init__()
        self.dim = dim
        self.set_type = str(set_type).lower()
        # z_m = φ([H_m; r_m; u_m])
        self.token_in = nn.Sequential(
            nn.Linear(dim + 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        if self.set_type == 'deepset':
            self.phi = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
            self.rho = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
        else:
            self.set_type = 'sab'
            n_layers = max(1, int(num_layers))
            self.sabs = nn.ModuleList([
                _SAB(dim, num_heads=num_heads, dropout=dropout)
                for _ in range(n_layers)
            ])
            self.rho = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.out_norm = nn.LayerNorm(dim)

    def _build_tokens(self, h_t, h_v, h_a, r_t, r_v, r_a, u_t, u_v, u_a):
        def pack(h, r, u):
            return torch.cat([h, r.unsqueeze(-1), u.unsqueeze(-1)], dim=-1)

        z_t = self.token_in(pack(h_t, r_t, u_t))
        z_v = self.token_in(pack(h_v, r_v, u_v))
        z_a = self.token_in(pack(h_a, r_a, u_a))
        # [M=3, N, D] — shared φ, no positional ids
        return torch.stack([z_t, z_v, z_a], dim=0)

    def forward(self, h_t, h_v, h_a, r_t, r_v, r_a, u_t, u_v, u_a):
        tokens = self._build_tokens(h_t, h_v, h_a, r_t, r_v, r_a, u_t, u_v, u_a)
        if self.set_type == 'deepset':
            # Deep Sets: ρ(sum φ(z_m)); sum is permutation-invariant
            pooled = self.phi(tokens).sum(dim=0)
            fused = self.rho(pooled)
        else:
            x = tokens
            for sab in self.sabs:
                x = sab(x)
            pooled = x.mean(dim=0)  # symmetric pool
            fused = self.rho(pooled)
        return self.out_norm(fused)
