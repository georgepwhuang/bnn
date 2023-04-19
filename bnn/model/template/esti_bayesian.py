from abc import ABC
from typing import List, Union, Optional

import copy
import torch
from torch.nn import functional as F

from bnn.model.template.bayesian import BayesianClassifier
from bnn.model.template.esti_naive import EstiNaiveClassifier


class EstiBayesianClassifier(BayesianClassifier, EstiNaiveClassifier, ABC):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], 
                 noisy_checkpoint: Optional[str] = None, *args, **kwargs):
        super(EstiBayesianClassifier, self).__init__(labels=labels, beta=beta, *args, **kwargs)
        self.noisy_checkpoint = noisy_checkpoint

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        with torch.no_grad():
            noisy_logits = F.softmax(self.noisy_model(data), -1)
            prev_logits = F.softmax(self.prev(data), -1)
            transition = torch.mul(self.confusion_matrix, noisy_logits.unsqueeze(2).div(prev_logits.unsqueeze(1)))  + 1e-16
            transition = transition / transition.sum(1, keepdim=True)
            transition = transition.nan_to_num(0, 0, 0)
            transition = (self.identity * (self.trainer.max_epochs - self.current_epoch) 
                          + transition * self.current_epoch)/self.trainer.max_epochs
            transition = transition / transition.sum(1, keepdim=True)
        logits = torch.bmm(transition, true_logits.unsqueeze(-1)).squeeze(-1) + 1e-16
        log_logits = torch.log(logits)
        celoss = F.nll_loss(log_logits, label)
        self.log('train_cross_entropy_loss', celoss)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss()
        self.log('train_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.train_dataloader), self.current_epoch,
                                            self.trainer.max_epochs) * klloss
        self.log('train_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        with torch.no_grad():
            noisy_logits = F.softmax(self.noisy_model(data), -1)
            prev_logits = F.softmax(self.prev(data), -1)
            transition = torch.mul(self.confusion_matrix, noisy_logits.unsqueeze(2).div(prev_logits.unsqueeze(1))) + 1e-16
            transition = transition / transition.sum(1, keepdim=True)
            transition = transition.nan_to_num(0, 0, 0)
            transition = (self.identity * (self.trainer.max_epochs - self.current_epoch) 
                          + transition * self.current_epoch)/self.trainer.max_epochs
            transition = transition / transition.sum(1, keepdim=True)
        logits = torch.bmm(transition, true_logits.unsqueeze(-1)).squeeze(-1) + 1e-16
        log_logits = torch.log(logits)
        celoss = F.nll_loss(log_logits, label)
        self.log('val_cross_entropy_loss', celoss, on_epoch=True)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss()
        self.log('val_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.val_dataloaders), self.current_epoch,
                                            self.trainer.max_epochs) * klloss
        self.log('val_loss', total_loss)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)

    def test_step(self, batch, batch_idx):
        data, corrected_label = batch
        output = self(data)
        celoss = F.cross_entropy(output, corrected_label)
        self.log('test_cross_entropy_loss', celoss)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss()
        self.log('test_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.test_dataloaders), self.current_epoch,
                                            self.trainer.max_epochs) * klloss
        self.log('test_loss', total_loss)
        logits = F.softmax(output, -1)
        self.test_metrics(logits, corrected_label)
        self.log_scalars(self.test_metrics)
