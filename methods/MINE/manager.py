import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from utils.metrics import AverageMeter, Metrics

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .loss import InfoNCEloss, CMD, DiffLoss

from thop import profile
from torchinfo import summary

__all__ = ['DUMIR']


class MINE:

    def __init__(self, args, data, model):

        self.logger = logging.getLogger(args.logger_name)
        self.writer = SummaryWriter(log_dir=args.pred_output_path)

        self.device, self.model = model.device, model.model

        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, verbose=True, patience=args.wait_patience)
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

        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _train(self, args):
    
        self.early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
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

                    preds = self.model(text_feats, video_feats, audio_feats)

                    text_feats_pro = preds['feat_t']
                    video_feats_pro = preds['feat_v']
                    audio_feats_pro = preds['feat_a']
                    feat_fusion_pro = preds['feat_fusion']

                    cons_loss = self.cons_criterion(text_feats_pro, audio_feats_pro, video_feats_pro)

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
                        loss = loss + self.diff_weight * diff_loss + self.sim_weight * cmd_loss
                    
                    cur_step = epoch * len(pbar) + step
                    self.writer.add_scalar("train/cls_loss", cls_loss.item(), cur_step)
                    self.writer.add_scalar("train/cons_loss", cons_loss.item(), cur_step)
                    if self.use_shared_private:
                        self.writer.add_scalar("train/diff_loss", diff_loss.item(), cur_step)
                        self.writer.add_scalar("train/cmd_loss", cmd_loss.item(), cur_step)
                    
                    total_logits = torch.cat((total_logits, preds['logits_fusion']))
                    total_labels = torch.cat((total_labels, label_ids))

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()

            # tensorboard on train
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            train_outputs = self.metrics(y_true, y_pred)
            self.writer.add_scalar("train/acc", train_outputs['acc'], epoch)
            
            # eval
            outputs = self._get_outputs(args, mode = 'eval')
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(outputs['loss'])
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"ReduceLROnPlateau: learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            eval_score = outputs[args.eval_monitor]
            
            # tensorboard on eval
            self.writer.add_scalar("eval/acc", outputs['acc'], epoch)
            self.writer.add_scalar("eval/cls_loss", outputs['cls_loss'], epoch)
            self.writer.add_scalar("eval/cons_loss", outputs['cons_loss'], epoch)
            self.writer.add_scalars("all_loss", {
                "train": round(loss_record.avg, 4),
                "val": outputs['loss'],
            }, global_step=epoch)


            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(self.early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4),
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            self.early_stopping(eval_score, self.model)
            
            # # test
            # self.logger.info("***** Epoch: %s: Test results *****", str(epoch + 1))
            # test_results = self._get_outputs(args, mode = 'test', show_results=True, show_matrix=False)
            # self.logger.info("***** Epoch: %s: Test End *****", str(epoch + 1))

            if self.early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = self.early_stopping.best_score
        self.model = self.early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)

    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False, show_matrix=False):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
            # self.model = self.early_stopping.best_model
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_features = torch.empty((0, self.model.model.d_t)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)

        loss_record = AverageMeter()
        cls_loss_record = AverageMeter()
        cons_loss_record = AverageMeter()


        for batch in tqdm(dataloader, desc="Iteration", leave=False,
                  bar_format='{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):


            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            sample_index = batch['sample_index']

            with torch.set_grad_enabled(False):

                preds = self.model(text_feats, video_feats, audio_feats)

                text_feats_pro = preds['feat_t']
                video_feats_pro = preds['feat_v']
                audio_feats_pro = preds['feat_a']
                feat_fusion_pro = preds['feat_fusion']

                total_logits = torch.cat((total_logits, preds['logits_fusion']))
                total_features = torch.cat((total_features, feat_fusion_pro))
                total_labels = torch.cat((total_labels, label_ids))

                cons_loss = self.cons_criterion(text_feats_pro, audio_feats_pro, video_feats_pro)

                loss_pred_fusion = self.criterion(preds['logits_fusion'], label_ids)
                loss_pred_text = self.criterion(preds['logits_t'], label_ids)
                loss_pred_video = self.criterion(preds['logits_v'], label_ids)
                loss_pred_audio = self.criterion(preds['logits_a'], label_ids)

                cls_loss = loss_pred_fusion + self.lamba*(loss_pred_text+loss_pred_audio+loss_pred_video)/3
                loss = cls_loss #+ self.lambc*cons_loss
                if self.use_shared_private:
                    diff_loss = self._get_diff_loss(preds)
                    cmd_loss = self._get_cmd_loss(preds)
                    loss = loss + self.diff_weight * diff_loss + self.sim_weight * cmd_loss

                loss_record.update(loss.item(), label_ids.size(0))
                cls_loss_record.update(cls_loss.item(), label_ids.size(0))
                cons_loss_record.update(cons_loss.item(), label_ids.size(0))


        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_feat = total_features.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results, show_matrix=show_matrix)
        outputs.update({'loss': loss_record.avg})
        outputs.update({'cls_loss': cls_loss_record.avg})
        outputs.update({'cons_loss': cons_loss_record.avg})

        if return_sample_results:

            outputs.update(
                {
                    'y_feat': y_feat,
                    'y_true': y_true,
                    'y_pred': y_pred
                }
            )

        return outputs


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

        test_results = self._get_outputs(args, mode = 'test', return_sample_results = True, show_results = True, show_matrix=True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)

        return test_results

