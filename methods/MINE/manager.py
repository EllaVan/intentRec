import copy
import logging

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from utils.functions import restore_model, save_model, EarlyStopping
from utils.metrics import AverageMeter, Metrics
from .loss import InfoNCEloss, CMD, DiffLoss

from thop import profile
from torchinfo import summary

__all__ = ['MINE']


class MINE:
    """
    Train / eval loop for the MINE backbone.

    Losses (when enabled):
      L = L_cls (+ unimodal) + diff_weight·L_diff + sim_weight·L_cmd
        + λ_R(t)·relation_loss_weight · (L_gate + per_mod + sign)

    Relation curriculum (single-stage schedule, not a two-stage model):
      t < Tw:          λ_R=0, β=0  (warmup; Intrinsic / main task only)
      Tw ≤ t < Tr:     λ_R ramps 0→1; β opens to relation_beta_init
      t ≥ Tr:          λ_R=1
    Optional EMA teacher supplies stop-grad LOO targets + sample confidence.
    """

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        self.writer = SummaryWriter(log_dir=args.pred_output_path)

        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=args.wait_patience)

        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']

        self.args = args
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.lamba = args.lamba
        self.lambc = args.lambc
        
        self.cons_criterion = InfoNCEloss(temperature=args.temperature)
        self.use_shared_private = getattr(args, 'use_shared_private', False)
        self.diff_weight = getattr(args, 'diff_weight', 0.1)
        self.sim_weight = getattr(args, 'sim_weight', 0.1)
        self.loss_diff = DiffLoss()
        self.loss_cmd = CMD()

        # ---- LOO relation importance loss (R̂ ↔ C_rel) ----
        self.use_relation_branch = getattr(args, 'use_relation_branch', False)
        self.relation_loss_weight = getattr(args, 'relation_loss_weight', 0.1)
        self.relation_criterion = nn.SmoothL1Loss(reduction='none')
        self.use_relation_sign_loss = getattr(args, 'use_relation_sign_loss', False)
        self.relation_sign_loss_weight = getattr(args, 'relation_sign_loss_weight', 0.1)
        self.relation_sign_eps = float(getattr(args, 'relation_sign_eps', 1e-2))
        self.relation_align_normalized = getattr(args, 'relation_align_normalized', True)
        self.relation_per_mod_weight = float(getattr(args, 'relation_per_mod_weight', 1.0))
        self.relation_beta_init = float(getattr(args, 'relation_beta_init', 0.0))
        self.relation_beta_warmup_epochs = int(getattr(args, 'relation_beta_warmup_epochs', 0))
        # Design §3: Tw / Tr for λ_R(t); default Tw = β-warmup
        self.relation_loss_warmup_epochs = int(getattr(
            args, 'relation_loss_warmup_epochs', self.relation_beta_warmup_epochs
        ))
        self.relation_loss_ramp_epochs = int(getattr(args, 'relation_loss_ramp_epochs', 10))
        # Do not early-stop before Tw+ramp; avoids half-trained relation ckpts.
        self.relation_es_min_epochs = int(getattr(
            args,
            'relation_es_min_epochs',
            self.relation_loss_warmup_epochs + self.relation_loss_ramp_epochs,
        ))
        # Only write ES best after β opens (discard Intrinsic-only peaks).
        self.relation_es_after_warmup = bool(
            getattr(args, 'relation_es_after_warmup', True)
        )
        self.relation_sup_conf = str(
            getattr(args, 'relation_sup_conf', 'none')
        ).lower()  # none | hard | soft
        self.relation_ema_decay = float(getattr(args, 'relation_ema_decay', 0.0))
        self._relation_lambda = 0.0
        self.ema_model = None
        if self.use_relation_branch and self.relation_ema_decay > 0:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            self.logger.info(
                'EMA teacher enabled for LOO targets (decay=%.4f)',
                self.relation_ema_decay,
            )

        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _train(self, args):
    
        self.early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            if self.use_relation_branch:
                opened = self._set_relation_curriculum_for_epoch(epoch)
                # First epoch with β open: clear Intrinsic-only ES state, then
                # start tracking. Do NOT reset again when λ_R opens (that would
                # discard the first post-β best from epoch Tw).
                if (
                    self.relation_es_after_warmup
                    and epoch == self.relation_beta_warmup_epochs
                ):
                    self.early_stopping.reset(keep_best_model=False)
                    self.logger.info(
                        'EarlyStopping reset at β open (epoch %d); '
                        'track best only with β>0 (patience=%d)',
                        epoch + 1, self.early_stopping.patience,
                    )
                elif opened:
                    self.early_stopping.counter = 0
                    self.early_stopping.early_stop = False
                    self.logger.info(
                        'EarlyStopping counter refreshed when λ_R opens '
                        '(keep best_score=%.4f, patience=%d)',
                        self.early_stopping.best_score,
                        self.early_stopping.patience,
                    )
            total_labels = torch.empty(0,dtype=torch.long).to(self.device)
            total_logits = torch.empty((0, args.num_labels)).to(self.device)
            total_preds = torch.empty(0,dtype=torch.long).to(self.device)
            self.model.train()
            loss_record = AverageMeter()

            pbar = tqdm(self.train_dataloader, desc="Iteration", 
                        bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            for step, batch in enumerate(pbar):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                sample_index = batch['sample_index']

                with torch.set_grad_enabled(True):

                    # LOO targets only when λ_R>0 (saves compute during warmup)
                    need_rel_sup = (
                        self.use_relation_branch and self._relation_lambda > 0.0
                    )
                    fwd_kwargs = {}
                    if need_rel_sup:
                        fwd_kwargs['label_ids'] = label_ids
                    preds = self.model(text_feats, video_feats, audio_feats, **fwd_kwargs)

                    # v1.3: InfoNCE probe uses private/unimodal feats in (t, a, v) order
                    # (cons_loss is logged but not added to the optim objective).
                    text_feats_pro = preds['feat_t']
                    video_feats_pro = preds['feat_v']
                    audio_feats_pro = preds['feat_a']
                    feat_fusion_pro = preds['feat_fusion']

                    cons_loss = self.cons_criterion(
                        text_feats_pro, audio_feats_pro, video_feats_pro
                    )

                    loss_pred_fusion = self.criterion(preds['logits_fusion'], label_ids)
                    loss_pred_text = self.criterion(preds['logits_t'], label_ids)
                    loss_pred_audio = self.criterion(preds['logits_a'], label_ids)
                    loss_pred_video = self.criterion(preds['logits_v'], label_ids)

                    cls_loss = loss_pred_fusion + self.lamba*(loss_pred_text+loss_pred_audio+loss_pred_video)/3
                    loss = cls_loss #+ self.lambc*cons_loss

                    diff_loss = preds['logits_fusion'].new_tensor(0.0)
                    cmd_loss = preds['logits_fusion'].new_tensor(0.0)
                    if self.use_shared_private:
                        diff_loss = self._get_diff_loss(preds)
                        cmd_loss = self._get_cmd_loss(preds)
                        # v1.3 objective: L_cls + λ_diff·L_diff + λ_sim·L_cmd
                        loss = loss + self.diff_weight * diff_loss + self.sim_weight * cmd_loss

                    relation_loss = preds['logits_fusion'].new_tensor(0.0)
                    relation_reg_loss = preds['logits_fusion'].new_tensor(0.0)
                    relation_sign_loss = preds['logits_fusion'].new_tensor(0.0)
                    if need_rel_sup:
                        teacher_out = self._teacher_loo_outputs(
                            text_feats, video_feats, audio_feats, label_ids
                        )
                        if teacher_out is not None:
                            for k in ('C_rel_t', 'C_rel_v', 'C_rel_a'):
                                preds[k] = teacher_out[k]
                            preds['rel_sup_logits'] = teacher_out['logits_fusion']
                        else:
                            preds['rel_sup_logits'] = preds['logits_fusion']
                        relation_loss, relation_reg_loss, relation_sign_loss = \
                            self._get_relation_loss(preds, label_ids)
                        loss = loss + (
                            self._relation_lambda
                            * self.relation_loss_weight
                            * relation_loss
                        )
                    
                    cur_step = epoch * len(pbar) + step
                    self.writer.add_scalar("train/cls_loss", cls_loss.item(), cur_step)
                    self.writer.add_scalar("train/cons_loss", cons_loss.item(), cur_step)
                    if self.use_shared_private:
                        self.writer.add_scalar("train/diff_loss", diff_loss.item(), cur_step)
                        self.writer.add_scalar("train/cmd_loss", cmd_loss.item(), cur_step)
                    if self.use_relation_branch:
                        self.writer.add_scalar(
                            "train/relation_lambda", self._relation_lambda, cur_step
                        )
                    if need_rel_sup:
                        self.writer.add_scalar("train/relation_loss", relation_loss.item(), cur_step)
                        self.writer.add_scalar("train/relation_reg_loss", relation_reg_loss.item(), cur_step)
                        if self.use_relation_sign_loss:
                            self.writer.add_scalar(
                                "train/relation_sign_loss", relation_sign_loss.item(), cur_step
                            )
                    
                    total_logits = torch.cat((total_logits, preds['logits_fusion']))
                    total_labels = torch.cat((total_labels, label_ids))

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()
                    self._update_ema_teacher()

            # tensorboard on train (epoch loop metrics)
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            train_outputs = self.metrics(y_true, y_pred)
            self.writer.add_scalar("train/acc", train_outputs['acc'], epoch)
            self.writer.add_scalar("train/f1", train_outputs['f1'], epoch)
            self.writer.add_scalar("train/loss", loss_record.avg, epoch)
            
            # eval (TensorBoard logged inside _get_outputs by mode)
            outputs = self._get_outputs(args, mode='eval', tb_step=epoch)
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(outputs['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"ReduceLROnPlateau: learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            eval_score = outputs[args.eval_monitor]

            self.writer.add_scalars("all_loss", {
                "train": round(loss_record.avg, 4),
                "val": outputs['loss'],
            }, global_step=epoch)
            self.writer.add_scalar("train/lr", new_lr, epoch)


            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(self.early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            # Warmup: still eval for TB/logs, but do not lock Intrinsic-only best.
            update_es = True
            if (
                self.use_relation_branch
                and self.relation_es_after_warmup
                and epoch < self.relation_beta_warmup_epochs
            ):
                update_es = False
            if update_es:
                self.early_stopping(eval_score, self.model)
            else:
                self.logger.info(
                    'Skip ES update during relation warmup (epoch %d < Tw=%d)',
                    epoch + 1, self.relation_beta_warmup_epochs,
                )

            # # test
            # self.logger.info("***** Epoch: %s: Test results *****", str(epoch + 1))
            # test_results = self._get_outputs(args, mode = 'test', show_results=True, show_matrix=False)
            # self.logger.info("***** Epoch: %s: Test End *****", str(epoch + 1))

            can_stop = True
            if self.use_relation_branch:
                # Let λ_R finish ramping before patience can terminate training.
                can_stop = (epoch + 1) >= self.relation_es_min_epochs
            if self.early_stopping.early_stop and can_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break
            if self.early_stopping.early_stop and not can_stop:
                self.early_stopping.early_stop = False
                self.early_stopping.counter = 0  # keep training through ramp
                self.logger.info(
                    'Defer EarlyStopping until epoch>=%d (curriculum min)',
                    self.relation_es_min_epochs,
                )

        self.best_eval_score = self.early_stopping.best_score
        if self.early_stopping.best_model is not None:
            self.model = self.early_stopping.best_model
        if self.use_relation_branch:
            # Keep β from the selected ckpt (do NOT force beta_init — that caused
            # train/test gate mismatch when best was mid-warmup).
            beta_now = self._current_relation_beta()
            self.logger.info(
                'Deploy ckpt relation β=%.4f (config beta_init=%.4f)',
                beta_now, self.relation_beta_init,
            )

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)

    def _get_outputs(self, args, mode='eval', return_sample_results=False,
                     show_results=False, show_matrix=False, tb_step=None):
        """
        Evaluate on train/eval/test. When tb_step is not None, write scalars to
        TensorBoard under the `{mode}/...` namespace.
        """
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader
        else:
            raise ValueError(f"Unknown mode={mode!r}; use train|eval|test")

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_features = torch.empty((0, self._fusion_net().d_t)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        loss_record = AverageMeter()
        cls_loss_record = AverageMeter()
        cons_loss_record = AverageMeter()
        diff_loss_record = AverageMeter()
        relation_loss_record = AverageMeter()

        for batch in tqdm(dataloader, desc="Iteration", leave=False,
                  bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):

                fwd_kwargs = {}
                if self.use_relation_branch:
                    fwd_kwargs['label_ids'] = label_ids
                preds = self.model(text_feats, video_feats, audio_feats, **fwd_kwargs)

                text_feats_pro = preds['feat_t']
                video_feats_pro = preds['feat_v']
                audio_feats_pro = preds['feat_a']
                feat_fusion_pro = preds['feat_fusion']

                total_logits = torch.cat((total_logits, preds['logits_fusion']))
                total_features = torch.cat((total_features, feat_fusion_pro))
                total_labels = torch.cat((total_labels, label_ids))

                cons_loss = self.cons_criterion(
                    text_feats_pro, audio_feats_pro, video_feats_pro
                )

                loss_pred_fusion = self.criterion(preds['logits_fusion'], label_ids)
                loss_pred_text = self.criterion(preds['logits_t'], label_ids)
                loss_pred_video = self.criterion(preds['logits_v'], label_ids)
                loss_pred_audio = self.criterion(preds['logits_a'], label_ids)

                cls_loss = loss_pred_fusion + self.lamba * (
                    loss_pred_text + loss_pred_audio + loss_pred_video
                ) / 3
                loss = cls_loss #+ self.lambc*cons_loss

                if self.use_shared_private:
                    diff_loss = self._get_diff_loss(preds)
                    cmd_loss = self._get_cmd_loss(preds)
                    loss = loss + self.diff_weight * diff_loss + self.sim_weight * cmd_loss
                    diff_loss_record.update(diff_loss.item(), label_ids.size(0))

                if self.use_relation_branch and self._relation_lambda > 0.0:
                    # eval uses student LOO (no EMA) for a cheap monitor signal
                    preds['rel_sup_logits'] = preds['logits_fusion']
                    relation_loss, _, _ = self._get_relation_loss(
                        preds, label_ids=label_ids
                    )
                    loss = loss + (
                        self._relation_lambda
                        * self.relation_loss_weight
                        * relation_loss
                    )
                    relation_loss_record.update(relation_loss.item(), label_ids.size(0))

                loss_record.update(loss.item(), label_ids.size(0))
                cls_loss_record.update(cls_loss.item(), label_ids.size(0))
                cons_loss_record.update(cons_loss.item(), label_ids.size(0))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        _, total_preds = total_probs.max(dim=1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_feat = total_features.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results, show_matrix=show_matrix)
        outputs.update({
            'loss': loss_record.avg,
            'cls_loss': cls_loss_record.avg,
            'cons_loss': cons_loss_record.avg,
        })
        if self.use_shared_private:
            outputs['diff_loss'] = diff_loss_record.avg
        if self.use_relation_branch:
            outputs['relation_loss'] = relation_loss_record.avg

        if tb_step is not None and self.writer is not None:
            self._log_outputs_to_tensorboard(mode, outputs, tb_step)

        if return_sample_results:
            outputs.update({
                'y_feat': y_feat,
                'y_true': y_true,
                'y_pred': y_pred,
            })

        return outputs

    def _log_outputs_to_tensorboard(self, mode, outputs, step):
        """Write metrics/losses from _get_outputs under `{mode}/...`."""
        prefix = str(mode)
        for key in ('acc', 'f1', 'prec', 'rec', 'loss', 'cls_loss', 'cons_loss'):
            if key in outputs:
                self.writer.add_scalar(f'{prefix}/{key}', outputs[key], step)
        if 'diff_loss' in outputs:
            self.writer.add_scalar(f'{prefix}/diff_loss', outputs['diff_loss'], step)
        if 'relation_loss' in outputs:
            self.writer.add_scalar(f'{prefix}/relation_loss', outputs['relation_loss'], step)


    def _fusion_net(self):
        model = self.model
        return model.model if hasattr(model, 'model') else model

    def _force_relation_beta(self, beta):
        fusion = self._fusion_net()
        if not hasattr(fusion, 'rel_beta'):
            return
        with torch.no_grad():
            fusion.rel_beta.fill_(float(beta))
            if self.ema_model is not None and hasattr(self.ema_model, 'rel_beta'):
                self.ema_model.rel_beta.fill_(float(beta))
        fusion.rel_beta.requires_grad = False

    def _current_relation_beta(self):
        fusion = self._fusion_net()
        if not hasattr(fusion, 'rel_beta'):
            return float('nan')
        return float(fusion.rel_beta.detach().cpu())

    def _relation_lambda_at(self, epoch):
        """λ_R(t): 0 before Tw, linear ramp Tw→Tr, then 1. Epoch is 0-based."""
        tw = self.relation_loss_warmup_epochs
        if epoch < tw:
            return 0.0
        if self.relation_loss_ramp_epochs <= 0:
            return 1.0
        tr = tw + self.relation_loss_ramp_epochs
        if epoch >= tr:
            return 1.0
        return float(epoch - tw) / float(self.relation_loss_ramp_epochs)

    def _set_relation_curriculum_for_epoch(self, epoch):
        """
        Single-stage schedule (design §3): β and λ_R warm up together.
        Returns True on the first epoch where λ_R becomes > 0.
        """
        if not self.use_relation_branch:
            return False
        prev_lambda = self._relation_lambda
        self._relation_lambda = self._relation_lambda_at(epoch)

        if epoch < self.relation_beta_warmup_epochs:
            beta = 0.0
        else:
            beta = self.relation_beta_init
        self._force_relation_beta(beta)

        opened = prev_lambda <= 0.0 and self._relation_lambda > 0.0
        if epoch == 0 or opened or ((epoch + 1) % 5 == 0):
            self.logger.info(
                'Relation curriculum epoch %d: λ_R=%.3f β=%.3f (Tw=%d ramp=%d)',
                epoch + 1, self._relation_lambda, beta,
                self.relation_loss_warmup_epochs, self.relation_loss_ramp_epochs,
            )
        return opened

    # backward-compatible alias
    def _set_relation_beta_for_epoch(self, epoch):
        return self._set_relation_curriculum_for_epoch(epoch)

    def _update_ema_teacher(self):
        if self.ema_model is None:
            return
        mu = self.relation_ema_decay
        with torch.no_grad():
            for p_t, p_s in zip(self.ema_model.parameters(), self.model.parameters()):
                p_t.data.mul_(mu).add_(p_s.data, alpha=1.0 - mu)

    def _teacher_loo_outputs(self, text_feats, video_feats, audio_feats, label_ids):
        """EMA teacher forward for stop-grad LOO targets; None if EMA disabled."""
        if self.ema_model is None:
            return None
        self.ema_model.eval()
        with torch.no_grad():
            return self.ema_model(
                text_feats, video_feats, audio_feats, label_ids=label_ids
            )

    @staticmethod
    def _modality_standardize(x):
        """Per-sample standardize over modality dim. x: [N, 3] -> [N, 3]."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (x - mean) / std

    @staticmethod
    def _batch_standardize(x):
        """Standardize a 1D batch vector."""
        return (x - x.mean()) / x.std().clamp_min(1e-6)

    def _sample_relation_confidence(self, logits, label_ids):
        """Sample weights c_i for L_rel (design §3.3). none|hard|soft."""
        if self.relation_sup_conf == 'hard':
            return (logits.argmax(dim=-1) == label_ids).to(dtype=logits.dtype)
        if self.relation_sup_conf == 'soft':
            probs = F.softmax(logits, dim=-1)
            return probs.gather(1, label_ids.long().unsqueeze(1)).squeeze(1)
        return logits.new_ones(logits.size(0))

    def _mean_smooth_l1(self, pred, tgt, sample_w):
        per = self.relation_criterion(pred, tgt)
        if per.dim() > 1:
            per = per.mean(dim=-1)
        w = sample_w / sample_w.sum().clamp_min(1e-8)
        return (w * per).sum()

    def _get_relation_loss(self, outputs, label_ids=None):
        """
        Supervise predicted R̂ against LOO target C_rel (both [N,3]).

          L_gate    = SmoothL1(mod-norm R̂, mod-norm C_rel)
          L_per_mod = mean_m SmoothL1(batch-z R̂_m, batch-z C_rel_m)
          L_sign    = BCEWithLogits(R̂, 1[C_rel>eps]) on |C_rel|>eps   (optional)

        Optional sample confidence from rel_sup_logits (EMA teacher preferred).
        Returns (total, L_gate+L_per_mod, L_sign).
        """
        preds = torch.stack(
            [outputs['C_hat_t'], outputs['C_hat_v'], outputs['C_hat_a']], dim=-1
        )
        tgts = torch.stack(
            [outputs['C_rel_t'], outputs['C_rel_v'], outputs['C_rel_a']], dim=-1
        )

        if label_ids is None:
            sample_w = preds.new_ones(preds.size(0))
        else:
            logits_for_conf = outputs.get('rel_sup_logits', outputs['logits_fusion'])
            sample_w = self._sample_relation_confidence(logits_for_conf, label_ids)
            if float(sample_w.sum()) < 1e-8:
                zero = preds.new_tensor(0.0)
                return zero, zero, zero

        if self.relation_align_normalized:
            gate_loss = self._mean_smooth_l1(
                self._modality_standardize(preds),
                self._modality_standardize(tgts),
                sample_w,
            )
        else:
            gate_loss = self._mean_smooth_l1(preds, tgts, sample_w)

        per_mod_loss = preds.new_tensor(0.0)
        if self.relation_per_mod_weight > 0:
            for i in range(3):
                per_mod_loss = per_mod_loss + self._mean_smooth_l1(
                    self._batch_standardize(preds[:, i]),
                    self._batch_standardize(tgts[:, i]),
                    sample_w,
                )
            per_mod_loss = per_mod_loss / 3.0

        rel_loss = gate_loss + self.relation_per_mod_weight * per_mod_loss

        sign_loss = preds.new_tensor(0.0)
        if self.use_relation_sign_loss:
            mask = tgts.abs() > self.relation_sign_eps
            if mask.any():
                w_elem = sample_w.unsqueeze(-1).expand_as(preds)[mask]
                target = (tgts[mask] > self.relation_sign_eps).to(dtype=preds.dtype)
                per = F.binary_cross_entropy_with_logits(
                    preds[mask], target, reduction='none'
                )
                sign_loss = (w_elem * per).sum() / w_elem.sum().clamp_min(1e-8)

        total = rel_loss
        if self.use_relation_sign_loss:
            total = total + self.relation_sign_loss_weight * sign_loss
        return total, rel_loss, sign_loss

    def _pool_sp_feat(self, feat):
        """CDPR uses sequence features [N,L,D]; MINE pooled features are already [N,D]."""
        if feat.dim() == 3:
            return feat.mean(dim=1)
        return feat

    def _get_cmd_loss(self, outputs):
        shared_t = self._pool_sp_feat(outputs['shared_t'])
        shared_v = self._pool_sp_feat(outputs['shared_v'])
        shared_a = self._pool_sp_feat(outputs['shared_a'])
        loss = self.loss_cmd(shared_t, shared_v, 5)
        loss += self.loss_cmd(shared_t, shared_a, 5)
        loss += self.loss_cmd(shared_a, shared_v, 5)
        return loss / 3.0

    def _get_diff_loss(self, outputs):
        shared_t = self._pool_sp_feat(outputs['shared_t'])
        shared_v = self._pool_sp_feat(outputs['shared_v'])
        shared_a = self._pool_sp_feat(outputs['shared_a'])
        private_t = self._pool_sp_feat(outputs['private_t'])
        private_v = self._pool_sp_feat(outputs['private_v'])
        private_a = self._pool_sp_feat(outputs['private_a'])

        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)
        return loss

    def _test(self, args):
        if self.use_relation_branch:
            # Use β baked into the selected/restored ckpt (same as training regime).
            self.logger.info(
                'Test with ckpt relation β=%.4f', self._current_relation_beta()
            )
        # Log final test metrics at step=0 under test/...
        test_results = self._get_outputs(
            args, mode='test', return_sample_results=True,
            show_results=True, show_matrix=True, tb_step=0,
        )
        test_results['best_eval_score'] = round(self.best_eval_score, 4)
        if self.writer is not None:
            self.writer.add_scalar('test/best_eval_score', self.best_eval_score, 0)
            self.writer.flush()

        return test_results

