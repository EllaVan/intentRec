import torch
from torch import nn
import torch.nn.functional as F

from ..SubNets.AlignNets import AlignSubNet
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder

class TimeSymmetricGate(nn.Module):
    def __init__(self, dim, alpha=5):
        super().__init__()
        # 共享打分头：每个时刻的 D 维特征压成标量分数
        self.fc = nn.Linear(dim, 1)
        # 共享不确定性映射：每个时刻的偏差压成标量
        self.unc_fc = nn.Linear(dim, 1)
        # self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha = alpha

    def _temporal_gate(self, feats):
        """
        对单一模态沿时间维 L 计算门控。
        feats: [L, N, D]
        return: [L, N]，沿 L 的 softmax 权重
        """
        # 每个时刻的内容打分（用在未池化的时间步上）
        raw_scores = self.fc(feats).squeeze(-1)  # [L, N]

        # 全局池化均值，作为该模态的参考中心
        g = feats.mean(dim=0, keepdim=True)  # [1, N, D]
        # 将原 var(dim=0) 落到每个 t：相对全局均值的平方偏差
        u = self.unc_fc((feats - g).pow(2)).squeeze(-1)  # [L, N]

        # 加入不确定性惩罚
        scores = raw_scores - self.alpha * u
        # 沿 L 竞争；再乘 L，使权重均值为 1，避免 1/L 把特征压得很小
        # 或者考虑用sigmoid
        gates = F.softmax(scores, dim=0) * feats.size(0)
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

    def _kl_div(self, p, log_p, log_q):
        """
        计算 KL(p || q)，沿类别维求和。
        p / log_p / log_q: [..., C]
        return: [...]
        """
        return (p * (log_p - log_q)).sum(dim=-1)

    def _cross_modal_disagreement(self, logits_t, logits_v, logits_a):
        """
        按跨模态预测分歧计算每个模态的 d：
        d_t = 1/2 * (KL(p_t || p_v) + KL(p_t || p_a))
        logits: [L, N, C]
        return: d_t, d_v, d_a，形状均为 [L, N]
        """
        p_t = F.softmax(logits_t, dim=-1)
        p_v = F.softmax(logits_v, dim=-1)
        p_a = F.softmax(logits_a, dim=-1)
        log_p_t = F.log_softmax(logits_t, dim=-1)
        log_p_v = F.log_softmax(logits_v, dim=-1)
        log_p_a = F.log_softmax(logits_a, dim=-1)

        d_t = 0.5 * (self._kl_div(p_t, log_p_t, log_p_v) + self._kl_div(p_t, log_p_t, log_p_a))
        d_v = 0.5 * (self._kl_div(p_v, log_p_v, log_p_t) + self._kl_div(p_v, log_p_v, log_p_a))
        d_a = 0.5 * (self._kl_div(p_a, log_p_a, log_p_t) + self._kl_div(p_a, log_p_a, log_p_v))
        return d_t, d_v, d_a

    def compute_emc(self, h_t, h_v, h_a, logits_t, logits_v, logits_a,
                         d_t, d_v, d_a):
        """
        能量引导的模态协同损失。
        E(m) = alpha * ||z_m||^2 + beta * ell_m + gamma * d_m
        L_EMC = L_gap + delta * Σ_m ||∂E(m)/∂z_m||^2
        h / logits: [L, N, D] / [L, N, C]，d: [L, N]
        """
        # if label_ids is None:
        #     return torch.zeros((), device=h_t.device)

        alpha = getattr(self.args, 'emc_alpha', 1.0)
        beta = getattr(self.args, 'emc_beta', 1.0)
        gamma = getattr(self.args, 'emc_gamma', 1.0)
        delta = getattr(self.args, 'emc_delta', 0.01)

        # 逐模态构造能量：特征范数、任务损失、跨模态分歧
        modalities = [(h_t, logits_t, d_t), (h_v, logits_v, d_v), (h_a, logits_a, d_a)]
        energies = []
        z_norms = []
        u_preds = []
        hs = []
        for h, logits, d_m in modalities:
            z_mean = h.mean(dim=0)                 # [N, D]
            z_norm = (z_mean ** 2).mean()
            
            # logits_mean = logits.mean(dim=0)       # [N, C]
            # l_m = F.cross_entropy(logits_mean, label_ids)
            logp = F.log_softmax(logits, dim=-1)  # logp = log(p)
            p = logp.exp()
            entropy = - (p * logp).sum(dim=-1)      # [B, T]
            logC = torch.log(torch.tensor(self.output_dim, dtype=logits_t.dtype, device=logits.device))
            u_pred = entropy / logC
            u_preds.append(u_pred)
            
            d_scalar = d_m.mean()
            E_m = alpha * z_norm + beta * u_pred + gamma * d_scalar
            energies.append(E_m)
            z_norms.append(z_norm)
            hs.append(h)

        # E_t, E_v, E_a = energies
        # # 三模态能量差的平方和
        # L_gap = (E_t - E_v) ** 2 + (E_t - E_a) ** 2 + (E_v - E_a) ** 2

        # # 仅对 ||z_m||^2 做梯度正则，避免对 CE/KL 求高阶导
        # grad_reg = torch.zeros((), device=h_t.device)
        # if torch.is_grad_enabled():
        #     for z_norm_m, h in zip(z_norms, hs):
        #         E_norm = alpha * z_norm_m
        #         grads = torch.autograd.grad(
        #             E_norm, h, create_graph=True, retain_graph=True, allow_unused=True
        #         )[0]
        #         if grads is not None:
        #             grad_reg = grad_reg + (grads ** 2).mean()

        # return L_gap + delta * grad_reg
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
        
        h_t = h_t[-1] #取最后一维做预测（有Conv1D）
        h_v = h_v[-1]
        h_a = h_a[-1]
        
        # 单模态预测
        logits_t = self.pred_t(h_t)
        logits_v = self.pred_v(h_v)
        logits_a = self.pred_a(h_a)

        # 跨模态预测分歧 d_m = 1/2 * Σ_{m'≠m} KL(p_m || p_{m'})
        d_t, d_v, d_a = self._cross_modal_disagreement(logits_t, logits_v, logits_a)
        
        # 样本级模态可靠性
        energies = self.compute_emc(h_t, h_v, h_a, logits_t, logits_v, logits_a, d_t, d_v, d_a)
        E_t, E_v, E_a = energies
        r_t = F.sigmoid(E_t)
        r_v = F.sigmoid(E_v)
        r_a = F.sigmoid(E_a)
        
        # 有符号类别证据
        sum_all_t = logits_t.sum(dim=-1, keepdim=True)   
        e_t = (self.output_dim / (self.output_dim - 1)) * logits_t - (1 / (self.output_dim - 1)) * sum_all_t
        sum_all_v = logits_v.sum(dim=-1, keepdim=True)
        e_v = (self.output_dim / (self.output_dim - 1)) * logits_v - (1 / (self.output_dim - 1)) * sum_all_v
        sum_all_a = logits_a.sum(dim=-1, keepdim=True)
        e_a = (self.output_dim / (self.output_dim - 1)) * logits_a - (1 / (self.output_dim - 1)) * sum_all_a
        
        # 模态融合
        feat_fusion = r_t.unsqueeze(-1) * h_t + r_v.unsqueeze(-1) * h_v + r_a.unsqueeze(-1) * h_a
        logits_fusion = self.pred_fusion(feat_fusion)
        logits_fusion = 0.5*logits_fusion + 0.5*(e_t + e_v + e_a)
        
        res = {
            'feat_t': h_t,
            'feat_v': h_v,
            'feat_a': h_a,
            'feat_fusion': feat_fusion,
            'logits_t':logits_t,
            'logits_v':logits_v,
            'logits_a':logits_a,
            'logits_fusion':logits_fusion,
            'gate_t':gate_t,
            'gate_v':gate_v,
            'gate_a':gate_a,
            'r_t':r_t,
            'r_v':r_v,
            'r_a':r_a,
        }
        
        return res