import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

__all__ = ['BERTEncoder', 'BottleneckMLP', 'LocalContentAttention']


class BottleneckMLP(nn.Module):
    """Bottleneck MLP: in_dim -> hidden_dim -> out_dim."""

    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0, activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LocalContentAttention(nn.Module):
    """
    Multi-head local-content attention.
    Query = local features, Key/Value = content (e.g. global context).

    Inputs use [L, N, D] (time, batch, dim) to match MINE temporal tensors.
    """

    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, local_feat, content_feat):
        """
        local_feat:   [L, N, D]
        content_feat: [L, N, D] or [1, N, D] / [N, D]
        return:
            out:  [L, N, D]
            attn: [N, H, L, S]
        """
        if content_feat.dim() == 2:
            content_feat = content_feat.unsqueeze(0)
        if content_feat.size(0) == 1 and local_feat.size(0) > 1:
            content_feat = content_feat.expand(local_feat.size(0), -1, -1)

        L, N, D = local_feat.shape
        S = content_feat.size(0)
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(local_feat)      # [L, N, D]
        k = self.k_proj(content_feat)    # [S, N, D]
        v = self.v_proj(content_feat)    # [S, N, D]

        # [N, H, L/S, d]
        q = q.permute(1, 0, 2).reshape(N, L, H, d).transpose(1, 2)
        k = k.permute(1, 0, 2).reshape(N, S, H, d).transpose(1, 2)
        v = v.permute(1, 0, 2).reshape(N, S, H, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [N, H, L, S]
        attn = self.dropout(attn.softmax(dim=-1))

        out = torch.matmul(attn, v)  # [N, H, L, d]
        out = out.transpose(1, 2).reshape(N, L, D).permute(1, 0, 2)  # [L, N, D]
        out = self.out_proj(out)
        out = self.norm(local_feat + out)

        return out, attn


class BERTEncoder(BertPreTrainedModel):

    # 在BERTEncoder类内部添加
    @property
    def all_tied_weights_keys(self):
        return self._tied_weights_keys


    def __init__(self, config):

        super(BERTEncoder, self).__init__(config)
        self.bert = BertModel(config)

    def forward(self, text_feats):

        input_ids, input_mask, segment_ids = text_feats[:, 0], text_feats[:, 1], text_feats[:, 2]
        outputs = self.bert(input_ids = input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states
