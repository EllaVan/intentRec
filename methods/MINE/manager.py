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
from .loss import InfoNCEloss


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

        self.metrics = Metrics(args)

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path)

    def _train(self, args):
    
        self.early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch",
                            bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
            total_labels = torch.empty(0,dtype=torch.long).to(self.device)
            total_logits = torch.empty((0, args.num_labels)).to(self.device)
            total_preds = torch.empty(0,dtype=torch.long).to(self.device)
            self.model.train()
            loss_record = AverageMeter()

            pbar = tqdm(self.train_dataloader, desc="Iteration", 
                        bar_format='{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
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

                    loss_fusion = self.criterion(preds['logits_fusion'], label_ids)
                    loss_text = self.criterion(preds['logits_t'], label_ids)
                    loss_audio = self.criterion(preds['logits_a'], label_ids)
                    loss_video = self.criterion(preds['logits_v'], label_ids)

                    cls_loss = loss_fusion + self.lamba*(loss_text+loss_audio+loss_video)/3
                    loss =  cls_loss #+ self.lambc*cons_loss
                    
                    cur_step = epoch * len(pbar) + step
                    self.writer.add_scalar("train/cls_loss", cls_loss.item(), cur_step)
                    self.writer.add_scalar("train/cons_loss", cons_loss.item(), cur_step)
                    
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

            if self.early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = self.early_stopping.best_score
        self.model = self.early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)

    def _get_outputs(self, args, mode = 'eval', return_sample_results = False, show_results = False):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
            self.model = self.early_stopping.best_model
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


        for batch in tqdm(dataloader, desc="Iteration"):

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

                loss_fusion = self.criterion(preds['logits_fusion'], label_ids)
                loss_text = self.criterion(preds['logits_t'], label_ids)
                loss_audio = self.criterion(preds['logits_v'], label_ids)
                loss_video = self.criterion(preds['logits_a'], label_ids)

                cls_loss = loss_fusion + self.lamba*(loss_text+loss_audio+loss_video)/3
                loss = cls_loss #+ self.lambc*cons_loss

                loss_record.update(loss.item(), label_ids.size(0))
                cls_loss_record.update(cls_loss.item(), label_ids.size(0))
                cons_loss_record.update(cons_loss.item(), label_ids.size(0))


        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_feat = total_features.cpu().numpy()

        outputs = self.metrics(y_true, y_pred, show_results=show_results)
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


    def _test(self, args):

        test_results = self._get_outputs(args, mode = 'test', return_sample_results = True, show_results = True)
        test_results['best_eval_score'] = round(self.best_eval_score, 4)

        return test_results

