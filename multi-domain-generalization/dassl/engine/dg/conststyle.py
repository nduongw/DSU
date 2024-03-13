from torch.nn import functional as F
import torch
import time
import datetime
import os.path as osp
from dassl.engine import *
from dassl.metrics import compute_accuracy
from dassl.utils import (
    MetricMeter, AverageMeter, count_num_param, load_checkpoint,
    save_checkpoint, load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.optim import build_optimizer, build_lr_scheduler
import torch.nn as nn

class ConstStyleModel(SimpleNet):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__(cfg, model_cfg, num_classes)
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            pertubration=model_cfg.BACKBONE.PERTUBATION,
            uncertainty=model_cfg.UNCERTAINTY,
            pos=model_cfg.POS,
            cfg=cfg,
            **kwargs
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim
    
    def forward(self, x, return_feature=False, store_feature=False, apply_conststyle=False):
        f = self.backbone(x, store_feature=store_feature, apply_conststyle=apply_conststyle)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y
    
@TRAINER_REGISTRY.register()
class ConstStyleTrainer(SimpleTrainer):
    """ConstStyle method."""
    
    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg
        print('Building model')
        self.model = ConstStyleModel(cfg, cfg.MODEL, self.num_classes)
        # print(self.model)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        # print('# params: {:,}'.format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('model', self.model, self.optim, self.sched)
    
    def train(self):
        """Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch(self.epoch)
            self.after_epoch()
            self.update_cluster()
        self.after_train()
        
    def before_epoch(self):
        for conststyle in self.model.backbone.conststyle:
            conststyle.clear_memory()
    
    def run_epoch(self, epoch):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, epoch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                    self.max_epoch - (self.epoch + 1)
                ) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar('train/' + name, meter.avg, n_iter)
                if self.args.wandb:
                    self.args.tracker.log({
                        f'training {name}': meter.avg 
                    }, step=self.epoch+1)
                    
            self.write_scalar('train/lr', self.get_current_lr(), n_iter)

            end = time.time()
    
    def update_cluster(self):
        if self.epoch == 0 or self.epoch % self.args.update_interval == 0:
            for idx, conststyle in enumerate(self.model.backbone.conststyle):
                conststyle.cal_mean_std(idx, self.epoch)
    
    def model_inference(self, input):
        if self.epoch == 0:
            return self.model(input)
        else:
            return self.model(input, store_feature=False, apply_conststyle=True)

    def forward_backward(self, batch, epoch):
        input, label = self.parse_batch_train(batch)
        if epoch == 0:
            output = self.model(input, store_feature=True, apply_conststyle=False)
        elif epoch % self.args.update_interval == 0 and epoch != 0:
            output = self.model(input, store_feature=True, apply_conststyle=True)
        else:
            output = self.model(input, store_feature=False, apply_conststyle=True)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'accuracy': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def save_model(self, epoch, directory, is_best=False):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()
            style_feats = {'mean': [], 'cov': [], 'std': []}
            
            for conststyle in self._models[name].backbone.conststyle:
                style_feats['mean'].append(conststyle.const_mean)
                style_feats['cov'].append(conststyle.const_cov)
                style_feats['std'].append(conststyle.const_std)

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    'state_dict': model_dict,
                    'epoch': epoch + 1,
                    'optimizer': optim_dict,
                    'scheduler': sched_dict,
                    'style_feats': style_feats
                },
                osp.join(directory, name),
                is_best=is_best
            )

    def load_model(self, directory, epoch=None):
        names = self.get_model_names()
        model_file = 'model.pth.tar-' + str(
            epoch
        ) if epoch else 'model-best.pth.tar'

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)
            
            for idx, conststyle in enumerate(self._models[name].backbone.conststyle):
                conststyle.const_mean = checkpoint['style_feats']['mean'][idx]
                conststyle.const_cov = checkpoint['style_feats']['cov'][idx]
                conststyle.const_std = checkpoint['style_feats']['std'][idx]
                