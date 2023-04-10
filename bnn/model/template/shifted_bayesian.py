from typing import List, Union

import torch
from torch.nn import functional as F

from bnn.model.template.bayesian import BayesianClassifier
from bnn.model.template.shifted_naive import ShiftedNaiveClassifier


class ShiftedBayesianClassifier(BayesianClassifier, ShiftedNaiveClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], correction_matrix: torch.Tensor,
                 beta: Union[str, float], *args, **kwargs):
        super(ShiftedBayesianClassifier, self).__init__(labels=labels, correction_matrix=correction_matrix, beta=beta,
                                                        *args, **kwargs)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        celoss = F.nll_loss(log_logits, label)
        self.log('train_cross_entropy_loss', celoss)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss(len(self.trainer.train_dataloader))
        self.log('train_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.train_dataloader), self.current_epoch,
                                            self.trainer.max_epochs) * klloss
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        celoss = F.nll_loss(log_logits, label)
        self.log('val_cross_entropy_loss', celoss, on_epoch=True)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss(len(self.trainer.val_dataloaders))
        self.log('val_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.val_dataloaders), self.current_epoch,
                                            self.trainer.max_epochs) * klloss
        self.log('val_total_loss', total_loss)
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
                klloss = klloss + module.kl_loss(len(self.trainer.test_dataloaders))
        self.log('test_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.test_dataloaders), self.current_epoch,
                                            self.trainer.max_epochs) * klloss
        self.log('test_total_loss', total_loss)
        logits = F.softmax(output, -1)
        self.test_metrics(logits, corrected_label)
        self.log_scalars(self.test_metrics)
