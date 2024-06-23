from torch.nn import functional as F
import torch
import numpy as np
import time
import os.path as osp
from dassl.engine import *
import csv
from dassl.metrics import compute_accuracy
from dassl.utils import (
    MetricMeter, AverageMeter, load_checkpoint,
    save_checkpoint, load_pretrained_weights
)
from dassl.optim import build_optimizer, build_lr_scheduler

def logarithmic_degradation(start, end, steps):
    x = np.linspace(1, steps, steps)
    return start - (start - end) * (np.log(x) / np.log(steps))

def linear_degradation(start, end, steps):
    return np.linspace(start, end, steps)

class ConstStyleModel(SimpleNet):
    def forward(self, x, domain, return_feature=False, store_feature=False, apply_conststyle=False, is_test=False):
        f = self.backbone(x, domain, store_feature=store_feature, apply_conststyle=apply_conststyle, is_test=is_test)
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
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
    
    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg
        print('Building model')
        self.model = ConstStyleModel(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)

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
        if self.args.dynamic_func == 'loga':
            momentum = logarithmic_degradation(0.01, self.cfg.TRAINER.RIDG.MOMENTUM, self.max_epoch)
            reg = logarithmic_degradation(0.1, self.cfg.TRAINER.RIDG.REG, self.max_epoch)
        elif self.args.dynamic_func == 'linear':
            momentum = linear_degradation(0.01, self.cfg.TRAINER.RIDG.MOMENTUM, self.max_epoch)
            reg = linear_degradation(0.1, self.cfg.TRAINER.RIDG.REG, self.max_epoch)
            
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            # loss_summary = self.forward_backward(batch, epoch, momentum[self.epoch], reg[self.epoch])
            loss_summary = self.forward_backward(batch, epoch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0:
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
    
    def model_inference(self, input, domain, is_test=False):
        if self.epoch == 0:
            return self.model(input, domain)
        else:
            return self.model(input, domain, store_feature=True, apply_conststyle=True, is_test=is_test)

    def forward_backward(self, batch, epoch):
        input, label, domain = self.parse_batch_train(batch)

        if epoch == 0:
            output = self.model(input, domain, store_feature=True, apply_conststyle=False)
        else:
            output = self.model(input, domain, store_feature=True, apply_conststyle=True)
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
        domain = batch['domain']
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain
    
    def save_model(self, epoch, directory, is_best=False):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()
            style_feats = {'mean': [], 'cov': []}
            
            for conststyle in self._models[name].backbone.conststyle:
                style_feats['mean'].append(conststyle.const_mean)
                style_feats['cov'].append(conststyle.const_cov)

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
    
    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()
        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            domain = torch.tensor([3 for i in range(len(label))])
            output = self.model_inference(input, domain, is_test=True)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            if self.args.wandb:
                self.args.tracker.log({
                    f'test {k}': v 
                }, step=self.epoch+1)
            self.write_scalar(tag, v, self.epoch)
    