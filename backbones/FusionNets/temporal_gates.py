"""Temporal (frame-level) gates used before multimodal pooling."""

import torch
from torch import nn
import torch.nn.functional as F


class ContextTemporalGate(nn.Module):
    """
    Per-modality temporal gate: score each frame with local+global context,
    then reweight the sequence (mean gate ≈ 1 via softmax * L).
    """

    def __init__(self, dim, hidden_dim=None, alpha=1.0, eps=1e-6):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2
        self.alpha = alpha
        self.eps = eps
        self.context_score = nn.Linear(dim, 1)
        self.local_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.context_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, feats):
        """
        Args:
            feats: [L, N, D]
        Returns:
            gated_feats: [L, N, D]
            gates: [L, N]  (sums to L over time)
        """
        L, N, D = feats.shape
        context_attn = F.softmax(self.context_score(feats).squeeze(-1), dim=0)
        context = (feats * context_attn.unsqueeze(-1)).sum(dim=0)  # [N, D]

        local_feature = self.local_proj(feats)
        content_feature = self.context_proj(context).unsqueeze(0).expand(L, -1, -1)
        scores = self.score(torch.relu(local_feature + content_feature)).squeeze(-1)
        gates = F.softmax(scores, dim=0) * L
        return feats * gates.unsqueeze(-1), gates
