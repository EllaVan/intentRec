import torch
from torch import nn
import torch.nn.functional as F

from ..SubNets.AlignNets import AlignSubNet
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
# 0.571M


class TimeSymmetricGate(nn.Module):
    def __init__(self, dim, alpha=1.0, eps=1e-6):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
        self.alpha = alpha
        self.eps = eps

    def _temporal_gate(self, feats):
        """
        feats: [L, N, D]
        return: [L, N]
        """
        L = feats.size(0)

        # 内容重要性
        raw_scores = self.fc(feats).squeeze(-1)  # [L, N]
        # 每个样本的时间中心
        center = feats.mean(dim=0, keepdim=True)  # [1, N, D]
        # 每个时间步偏离样本时间中心的程度
        deviation = (feats - center).pow(2).mean(dim=-1)  # [L, N]

        # 每个样本内部归一化，避免不同样本的特征尺度直接影响门控
        dev_mean = deviation.mean(dim=0, keepdim=True)
        dev_std = deviation.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(self.eps)

        deviation_norm = (deviation - dev_mean) / dev_std
        scores = raw_scores - self.alpha * deviation_norm

        # 均值为 1，便于保留特征整体尺度
        gates = F.softmax(scores, dim=0) * L
        return gates
    
    def forward(self, feats_t, feats_v, feats_a):
        """
        feats_x: [L, N, D]
        return: 三个模态各自的时序门控，形状均为 [L, N]
        """
        # 三模态共享参数，各自沿自身 L 做 softmax
        return (
            self._temporal_gate(feats_t),
            self._temporal_gate(feats_v),
            self._temporal_gate(feats_a),
        )

class MINE(nn.Module):

    def __init__(self, args):
        super(MINE, self).__init__()
        self.args = args
        
        self.text_subnet = BERTEncoder.from_pretrained(args.bert_path + args.text_backbone, cache_dir = args.cache_path)

        if args.need_aligned:
            self.alignNet = AlignSubNet(args, args.aligned_method)

        video_feat_dim = args.video_feat_dim
        text_feat_dim = args.text_feat_dim
        audio_feat_dim = args.audio_feat_dim
        dst_feature_dims = args.dst_feature_dims
        self.orig_d_t, self.orig_d_a, self.orig_d_v = text_feat_dim, audio_feat_dim, video_feat_dim
        self.d_t = self.d_a = self.d_v = dst_feature_dims
        self.output_dim = args.num_labels
        
        self.pred_t = nn.Linear(self.d_t, self.output_dim)
        self.pred_v = nn.Linear(self.d_v, self.output_dim)
        self.pred_a = nn.Linear(self.d_a, self.output_dim)
        self.pred_fusion = nn.Linear(dst_feature_dims, self.output_dim)
        
        self.proj_t = nn.Conv1d(self.orig_d_t, self.d_t, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        
        self.la = TimeSymmetricGate(self.d_t)

        self.num_heads = args.nheads
        self.layers = args.n_levels

        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v

        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

    
    def _js_divergence(self, logits_p, logits_q):
        """
        logits_p, logits_q: [L, N, C]
        return: [L, N]
        """
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        p = log_p.exp()
        q = log_q.exp()

        # log((p + q) / 2)，使用 logsumexp 保持数值稳定
        log_mix = torch.logaddexp(log_p, log_q) - torch.log(logits_p.new_tensor(2.0))
        mix = log_mix.exp()

        js = 0.5 * (p * (log_p - log_mix)).sum(dim=-1) + 0.5 * (q * (log_q - log_mix)).sum(dim=-1)

        # 理论上 JSD >= 0；消除可能出现的微小负数
        return js.clamp_min(0.0)


    def _cross_modal_disagreement(self, logits_t, logits_v, logits_a):
        """
        logits: [L, N, C]
        return: d_t, d_v, d_a, each [L, N]
        """
        d_t = 0.5 * (
            self._js_divergence(logits_t, logits_v)
            + self._js_divergence(logits_t, logits_a)
        )

        d_v = 0.5 * (
            self._js_divergence(logits_v, logits_t)
            + self._js_divergence(logits_v, logits_a)
        )

        d_a = 0.5 * (
            self._js_divergence(logits_a, logits_t)
            + self._js_divergence(logits_a, logits_v)
        )

        # 自然情况下 d_t、d_v、d_a 对称等价，
        # 这里保留逐模态接口，便于后续扩展。
        return d_t, d_v, d_a


    
    def compute_emc(
        self, h_t, h_v, h_a,
        logits_t, logits_v, logits_a,
        d_t, d_v, d_a
    ):
        """
        h:      [L, N, D]
        logits: [L, N, C]
        d:      [L, N]

        return:
            energies: list of 3 tensors, each [N]
        """
        alpha = getattr(self.args, "emc_alpha", 0.0)
        beta = getattr(self.args, "emc_beta", 1.0)
        gamma = getattr(self.args, "emc_gamma", 1.0)

        modalities = [
            (h_t, logits_t, d_t),
            (h_v, logits_v, d_v),
            (h_a, logits_a, d_a),
        ]

        energies = []

        for h, logits, d_m in modalities:
            # 1. Sample-level feature energy
            # h: [L, N, D]
            z_mean = h.mean(dim=0)                  # [N, D]
            z_norm = z_mean.pow(2).mean(dim=-1)     # [N]

            # 2. Predictive entropy
            logp = F.log_softmax(logits, dim=-1)    # [L, N, C]
            p = logp.exp()

            entropy = -(p * logp).sum(dim=-1)       # [L, N]
            logC = logits.new_tensor(logits.size(-1)).log()

            u_pred = entropy / logC                 # [L, N]
            u_pred = u_pred.mean(dim=0)             # [N]

            # 3. Sample-level disagreement
            d_scalar = d_m.mean(dim=0)              # [N]

            # Feature norm is optional; it is not intrinsically uncertainty.
            # log1p reduces scale sensitivity.
            feature_energy = torch.log1p(z_norm)

            E_m = (
                alpha * feature_energy
                + beta * u_pred
                + gamma * d_scalar
            )                                       # [N]

            energies.append(E_m)

        return energies


    def forward(self, text_data, video_data, audio_data, label_ids=None):
        """
        x_data: [N,L,D]
        """
        text_feats = self.text_subnet(text_data)
        
        if self.args.need_aligned:
            x_t, x_v, x_a  = self.alignNet(text_feats, video_data, audio_data)
        else:
            x_t, x_v, x_a  = text_feats, video_data, audio_data
            
        x_t = F.dropout(x_t.transpose(1, 2), p=self.text_dropout, training=self.training) #[N, D, L]
        x_v = x_v.transpose(1, 2)
        x_a = x_a.transpose(1, 2)
        
        proj_x_t = x_t if self.orig_d_t == self.d_t else self.proj_t(x_t)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        
        proj_x_t = proj_x_t.permute(2, 0, 1) #[L,N,D]
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        
        gate_t, gate_v, gate_a = self.la(proj_x_t, proj_x_v, proj_x_a)  # [L, N]
        # 广播到 [L, N, 1]，与投影特征 [L, N, D] 相乘
        gate_t = gate_t.unsqueeze(-1)
        gate_v = gate_v.unsqueeze(-1)
        gate_a = gate_a.unsqueeze(-1)
        # 时序门控加权（使用投影后的特征）
        h_t = proj_x_t * gate_t
        h_v = proj_x_v * gate_v
        h_a = proj_x_a * gate_a
        
        # h_t = h_t[-1] #取最后一维做预测（有Conv1D）
        # h_v = h_v[-1]
        # h_a = h_a[-1]
        
        # 单模态预测
        # logits_t = self.pred_t(h_t)
        # logits_v = self.pred_v(h_v)
        # logits_a = self.pred_a(h_a)

        # 跨模态预测分歧 d_m = 1/2 * Σ_{m'≠m} KL(p_m || p_{m'})
        
        # 单模态逐时间步预测
        # logits_t_seq = self.pred_t(h_t)  # [L_t, N, C]
        # logits_v_seq = self.pred_v(h_v)  # [L_v, N, C]
        # logits_a_seq = self.pred_a(h_a)  # [L_a, N, C]

        # 跨模态逐时间步分歧
        # 仅当三个模态的时间位置有对应关系时，才直接使用此计算。
        # 若 L_t、L_v、L_a 不相等或没有时间对齐，应改为池化后计算。
        # if (
        #     logits_t_seq.size(0) == logits_v_seq.size(0)
        #     and logits_t_seq.size(0) == logits_a_seq.size(0)
        # ):
        #     d_t, d_v, d_a = self._cross_modal_disagreement(
        #         logits_t_seq, logits_v_seq, logits_a_seq
        #     )
        # else:
        
        # 1. 每个模态独立时间池化
        h_t_pool = h_t.mean(dim=0)  # [N, D]
        h_v_pool = h_v.mean(dim=0)  # [N, D]
        h_a_pool = h_a.mean(dim=0)  # [N, D]

        # 2. 样本级单模态 logits
        logits_t = self.pred_t(h_t_pool)  # [N, C]
        logits_v = self.pred_v(h_v_pool)  # [N, C]
        logits_a = self.pred_a(h_a_pool)  # [N, C]

        # 3. 将 [N, C] 临时扩展为 [1, N, C]，
        # 复用已有 JSD 函数，得到样本级分歧 [N]
        d_t, d_v, d_a = self._cross_modal_disagreement(
            logits_t.unsqueeze(0),
            logits_v.unsqueeze(0),
            logits_a.unsqueeze(0),
        )

        d_t = d_t.squeeze(0)
        d_v = d_v.squeeze(0)
        d_a = d_a.squeeze(0)

        # 4. 样本级可靠性
        energies = self.compute_emc(
            h_t, h_v, h_a,
            logits_t.unsqueeze(0),
            logits_v.unsqueeze(0),
            logits_a.unsqueeze(0),
            d_t.unsqueeze(0),
            d_v.unsqueeze(0),
            d_a.unsqueeze(0),
        )

        E_t, E_v, E_a = energies

        # 高熵、高冲突对应更高能量时，能量越大，可靠性越低
        r_t = torch.sigmoid(-E_t)  # [N]
        r_v = torch.sigmoid(-E_v)  # [N]
        r_a = torch.sigmoid(-E_a)  # [N]

        # 5. 有符号类别证据
        def signed_evidence(logits):
            C = logits.size(-1)
            if C <= 1:
                raise ValueError("num_labels must be greater than 1")
            return C / (C - 1) * (
                logits - logits.mean(dim=-1, keepdim=True)
            )

        e_t = signed_evidence(logits_t)  # [N, C]
        e_v = signed_evidence(logits_v)
        e_a = signed_evidence(logits_a)

        # 6. 样本级特征融合
        feat_fusion = (
            r_t.unsqueeze(-1) * h_t_pool
            + r_v.unsqueeze(-1) * h_v_pool
            + r_a.unsqueeze(-1) * h_a_pool
        )  # [N, D]

        logits_joint = self.pred_fusion(feat_fusion)  # [N, C]

        # 不直接固定 0.5/0.5；证据分支单独设置较小的初始影响
        evidence_scale = getattr(self.args, "evidence_scale", 0.1)

        logits_fusion = 0.5*logits_joint + 0.5 * (
            e_t + e_v + e_a
        )  # [N, C]

        res = {
            "feat_t": h_t_pool,
            "feat_v": h_v_pool,
            "feat_a": h_a_pool,
            "feat_fusion": feat_fusion,
            "logits_t": logits_t,
            "logits_v": logits_v,
            "logits_a": logits_a,
            "logits_fusion": logits_fusion,
            "gate_t": gate_t,
            "gate_v": gate_v,
            "gate_a": gate_a,
            "r_t": r_t,
            "r_v": r_v,
            "r_a": r_a,
            "energy_t": E_t,
            "energy_v": E_v,
            "energy_a": E_a,
        }

        return res
