from torch.nn import functional as F
import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy


@TRAINER_REGISTRY.register()
class RIDG(TrainerX):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)
        self.check_cfg(cfg)
        self.init = torch.ones(self.num_classes, device=self.device)
        self.rational_bank = torch.zeros(self.num_classes, self.num_classes, self.model.backbone.out_features, device=self.device)
    
    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        features = self.model.backbone(input)
        output = self.model(input)
        
        rational = torch.zeros(self.num_classes, input.shape[0], self.model.backbone.out_features, device=self.device)
        for i in range(self.num_classes):
            rational[i] = (self.model.classifier.weight[i] * features)

        classes = torch.unique(label)
        loss_rational = 0.0
        for i in range(classes.shape[0]):
            rational_mean = rational[:, label==classes[i]].mean(dim=1)
            if self.init[classes[i]]:
                self.rational_bank[classes[i]] = rational_mean
                self.init[classes[i]] = False
            else:
                self.rational_bank[classes[i]] = (1 - self.cfg.TRAINER.RIDG.MOMENTUM) * self.rational_bank[classes[i]] + self.cfg.TRAINER.RIDG.MOMENTUM * rational_mean
            loss_rational += ((rational[:, label==classes[i]] - (self.rational_bank[classes[i]].unsqueeze(1)).detach())**2).sum(dim=2).mean()

        loss = F.cross_entropy(output, label) + self.cfg.TRAINER.RIDG.REG * loss_rational
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