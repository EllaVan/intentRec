class Param():

    def __init__(self, args):
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
        padding_mode / padding_loc / need_aligned / eval_monitor /
        train|eval|test_batch_size / wait_patience
        """
        return {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
            'eval_monitor': 'f1',
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            # Longer patience: curriculum needs Tw+ramp before ES can fire.
            'wait_patience': 12,
        }

    def _get_hyper_parameters(self, args):
        """
        v1.3 backbone + relation importance (design: 261112关系重要性).

        Keep use_relation_branch=False to reproduce v1.3.
        Set True for the optimized single-stage relation scheme:
          q=Ws s+Wp p, sp_concat fusion, signed_exp gate,
          Tw/Tr λ_R curriculum, EMA LOO teacher, soft conf L_rel.
        """
        return {
            # ---- backbone (aligned with save/v1.3) ----
            'num_train_epochs': 100,
            'dst_feature_dims': 360,
            'nheads': 4,
            'n_levels': 4,
            'attn_dropout': 0.0,
            'attn_dropout_v': 0.2,
            'attn_dropout_a': 0.2,
            'relu_dropout': 0.0,
            'embed_dropout': 0.1,
            'res_dropout': 0.0,
            'output_dropout': 0.2,
            'text_dropout': 0.4,
            'grad_clip': 0.5,
            'attn_mask': True,
            'conv1d_kernel_size_l': 1,
            'conv1d_kernel_size_v': 1,
            'conv1d_kernel_size_a': 1,
            'lr': 0.00003,
            'temperature': 0.05,
            'label_smoothing': 0.02,
            'lambc': 0.03,
            'lamba': 0.8,

            # ---- shared / private (v1.3) ----
            'use_shared_private': True,
            'sp_dropout': 0.2,
            'diff_weight': 0.1,
            'sim_weight': 0.1,

            # ---- reliability / evidence (v1.3) ----
            'reliability_mode': 'sigmoid',
            'use_evidence_fusion': True,

            # ---- fusion backend ----
            # weighted (v1.3 / relation-off) | tensor | set | cdpr
            'fusion_mode': 'weighted',
            'fusion_dropout': 0.2,
            'tensor_use_triple': True,
            'tensor_share_unary': False,
            'set_fusion_type': 'sab',
            'set_num_heads': 4,
            'set_num_layers': 1,
            'fusion_use_relation_gate': False,

            # ---- relation branch (design default when enabled) ----
            # False → exact v1.3 path; True → optimized relation scheme
            # (this file: relation ON)
            'use_relation_branch': True,
            'compute_rel_contrib': False,
            'pre_analysis_probe_epochs': 20,
            'pre_analysis_probe_lr': 0.001,

            # Descriptor / fusion (design §2)
            'relation_desc_mode': 'linear',       # linear: q=Ws s+Wp p | mlp: legacy Z
            'relation_fusion_mode': 'sp_concat',  # sp_concat: MLP([S;P]) | weighted
            'relation_content_mode': 'private',   # only for fusion_mode=weighted
            'relation_z_residual': 0.0,

            # Gate (design §4): signed_exp keeps negative R̂ as suppress
            # multiplicative | additive | softmax_mix | signed_exp
            'relation_gate_mode': 'signed_exp',
            'relation_alpha_init': 1.0,
            'relation_alpha_fixed': True,
            'relation_beta_init': 0.2,            # milder gate vs 0.5 (seed variance)
            'relation_beta_fixed': True,
            'relation_signed_exp_clip': 2.0,
            'relation_gate_renorm': True,
            'relation_conf_gate': False,          # used by multiplicative/additive only
            'relation_conf_scale': 2.0,
            'relation_conf_bias': 0.5,

            # Curriculum Tw / Tr (design §3); β opens with Tw
            'relation_beta_warmup_epochs': 10,    # Tw: β=0
            'relation_loss_warmup_epochs': 10,    # Tw: λ_R=0
            'relation_loss_ramp_epochs': 10,      # Tw→Tr linear λ_R
            # ES: skip warmup best; no stop before Tw+ramp; test keeps ckpt β
            'relation_es_after_warmup': True,
            'relation_es_min_epochs': 20,

            # EMA teacher for LOO targets (0 disables; still single-stage)
            'relation_ema_decay': 0.999,
            # Sample confidence on L_rel: none | hard | soft
            'relation_sup_conf': 'soft',

            # Loss weights
            'relation_loss_weight': 0.2,
            'relation_align_normalized': True,
            'relation_per_mod_weight': 1.0,
            'use_relation_sign_loss': True,
            'relation_sign_loss_weight': 0.4,
            'relation_sign_eps': 1e-2,
            'relation_loo_use_gate': False,
            'relation_use_fusion_norm': False,
            'relation_dropout': 0.2,
        }
