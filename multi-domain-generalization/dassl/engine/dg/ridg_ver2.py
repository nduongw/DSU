from torch.nn import functional as F
import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import save_checkpoint
import os.path as osp
import time
import csv
import numpy as np
from dassl.utils import (
    MetricMeter, AverageMeter
)

def logarithmic_degradation(start, end, steps):
    x = np.linspace(1, steps, steps)
    return start - (start - end) * (np.log(x) / np.log(steps))

def linear_degradation(start, end, steps):
    return np.linspace(start, end, steps)

@TRAINER_REGISTRY.register()
class ValueBasedRIDG(TrainerX):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.check_cfg(cfg)
        self.init = torch.ones(self.num_classes).to(self.device)
        self.rational_bank = torch.zeros(self.num_classes, self.num_classes, self.model.backbone.out_features).to(self.device)
        self.train_file_name = osp.join(cfg.OUTPUT_DIR, 'train_rational_value.csv')
        self.test_file_name = osp.join(cfg.OUTPUT_DIR, 'test_rational_value.csv')
        f_train = open(self.train_file_name, 'w')
        f_test = open(self.test_file_name, 'w')
        csvwriter_train = csv.writer(f_train)
        csvwriter_test = csv.writer(f_test)
        write_train_data = ['epoch', 'class', 'hindex1', 'hindex2', 'hindex3', 'hindex4', 'hindex5', 'value1', 'value2', 'value3', 'value4', 'value5']
        write_test_data = ['epoch', 'class', 'hindex1', 'hindex2', 'hindex3', 'hindex4', 'hindex5', 'value1', 'value2', 'value3', 'value4', 'value5','predicted_class']
        csvwriter_train.writerow(write_train_data)
        csvwriter_test.writerow(write_test_data)
        f_train.close()
        f_test.close()
        
    def forward_backward(self, batch, momentum, reg):
        input, label = self.parse_batch_train(batch)
        features = self.model.backbone(input)
        output = self.model(input)
        
        rational = torch.zeros(self.num_classes, input.shape[0], self.model.backbone.out_features).to(self.device)
        for i in range(self.num_classes):
            rational[i] = (self.model.classifier.weight[i] * features)

        classes = torch.unique(label)
        
        
        loss_rational = 0.0
        for i in range(classes.shape[0]):
            core_rational = torch.zeros(self.num_classes, self.model.backbone.out_features).to(self.device)
            class_rational = rational[:, label==classes[i]]
            for j in range(self.num_classes):
                s_rational = class_rational[j]
                if j == classes[i]:
                    argmax = torch.argmax(s_rational, dim=1)
                    for idx, val in enumerate(argmax):
                        if core_rational[j][val] == 0 or core_rational[j][val] < s_rational[idx][val]:
                            core_rational[j][val] = s_rational[idx][val]
                else:
                    argmin = torch.argmin(s_rational, dim=1)
                    for idx, val in enumerate(argmin):
                        if core_rational[j][val] == 0 or core_rational[j][val] > s_rational[idx][val]:
                            core_rational[j][val] = s_rational[idx][val]

            rational_mean = class_rational.mean(dim=1)
            merged_rational = torch.where(core_rational != 0, core_rational, rational_mean)
            if self.init[classes[i]]:
                self.rational_bank[classes[i]] = merged_rational
                self.init[classes[i]] = False
            else:
                self.rational_bank[classes[i]] = (1 - momentum) * self.rational_bank[classes[i]] + momentum * merged_rational
            loss_rational += ((rational[:, label==classes[i]] - (self.rational_bank[classes[i]].unsqueeze(1)).detach())**2).sum(dim=2).mean()
        ce_loss = F.cross_entropy(output, label)
        loss = ce_loss + reg * loss_rational
        print(f'cross entropy loss: {ce_loss} | rational loss: {reg * loss_rational}')
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

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()
            
            rational_bank = self.rational_bank

            save_checkpoint(
                {
                    'state_dict': model_dict,
                    'epoch': epoch + 1,
                    'optimizer': optim_dict,
                    'scheduler': sched_dict,
                    'rational_bank': rational_bank
                },
                osp.join(directory, name),
                is_best=is_best
            )
    
    def run_epoch(self):
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

        f_train = open(self.train_file_name, 'a')
        csvwriter_train = csv.writer(f_train)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, momentum[self.epoch], reg[self.epoch])
            batch_time.update(time.time() - end)
            losses.update(loss_summary)
            
            for i in range(self.num_classes):
                write_data = [self.epoch]
                write_value = self.rational_bank[i][i]
                write_value = write_value.detach().cpu().numpy()
                h_idx = np.argpartition(write_value, -5)[-5:]
                h_value = write_value[h_idx]
                write_data.append(i)
                write_data.extend(h_idx)
                write_data.extend(h_value)
                csvwriter_train.writerow(write_data)
                
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
        
        f_train.close()
    
    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        split = self.cfg.TEST.SPLIT
        print('Do evaluation on {} set'.format(split))
        data_loader = self.val_loader if split == 'val' else self.test_loader
        assert data_loader is not None
        
        f_test = open(self.test_file_name, 'a')
        csvwriter_test = csv.writer(f_test)

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            features = self.model.backbone(input)
            rand_idx = np.random.randint(0, label.shape[0])
            write_value = (self.model.classifier.weight[label[rand_idx].item()] * features[rand_idx])
            write_data = [self.epoch]
            write_value = write_value.detach().cpu().numpy()
            h_idx = np.argpartition(write_value, -5)[-5:]
            h_value = write_value[h_idx]
            write_data.append(label[rand_idx].item())
            write_data.extend(h_idx)
            write_data.extend(h_value)
            output = self.model_inference(input)
            predicted_class = torch.argmax(output[rand_idx]).item()
            write_data.append(predicted_class)
            csvwriter_test.writerow(write_data)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)
        
        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            if self.args.wandb:
                self.args.tracker.log({
                    f'test {k}': v 
                }, step=self.epoch+1)
            self.write_scalar(tag, v, self.epoch)