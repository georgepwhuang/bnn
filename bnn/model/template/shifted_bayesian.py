from typing import List, Union
import torch

from torch.nn import functional as F

from bnn.model.template.bayesian import BayesianClassifier
from bnn.model.template.shifted_naive import ShiftedNaiveClassifier


class ShiftedBayesianClassifier(BayesianClassifier, ShiftedNaiveClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], dataset_name: str,  beta: Union[str, float], *args, **kwargs):
        super(ShiftedBayesianClassifier, self).__init__(labels=labels, dataset_name=dataset_name, beta=beta, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        celoss = F.nll_loss(log_logits, label)
        with torch.no_grad():
            true_loss = F.cross_entropy(output, corrected_label)
        self.log('train_cross_entropy_loss', celoss)
        self.log('train_true_loss', true_loss)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss()
        self.log('train_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.train_dataloader), self.current_epoch, self.trainer.max_epochs) * klloss
        self.log('train_total_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        celoss = F.nll_loss(log_logits, label)
        with torch.no_grad():
            true_loss = F.cross_entropy(output, corrected_label)
        self.log('val_cross_entropy_loss', celoss,  on_epoch=True)
        self.log('val_true_loss', true_loss,  on_epoch=True)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss()
        self.log('val_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.val_dataloaders), self.current_epoch, self.trainer.max_epochs) * klloss
        self.log('val_total_loss', total_loss)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)
        self.val_true_metrics(true_logits, corrected_label)
        self.log_scalars(self.val_true_metrics)
        
    def test_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        celoss = F.nll_loss(log_logits, label)
        with torch.no_grad():
            true_loss = F.cross_entropy(output, corrected_label)
        self.log('test_cross_entropy_loss', celoss, on_epoch=True)
        self.log('test_true_loss', true_loss,  on_epoch=True)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss()
        self.log('test_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.test_dataloaders), self.current_epoch, self.trainer.max_epochs) * klloss
        self.log('test_total_loss', total_loss)
        logits = F.softmax(output, -1)
        self.test_metrics(logits, label)
        self.log_scalars(self.test_metrics)
        self.test_true_metrics(true_logits, corrected_label)
        self.log_scalars(self.test_true_metrics)