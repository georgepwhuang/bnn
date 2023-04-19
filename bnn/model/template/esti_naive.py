from abc import ABC
from typing import List, Union, Optional

import torch
import copy
from torch.nn import functional as F

from bnn.model.template.base import BaseClassifier


class EstiNaiveClassifier(BaseClassifier, ABC):
    def __init__(self, labels: Union[List[Union[str, int]], int], noisy_checkpoint: Optional[str] = None, *args, **kwargs):
        super(EstiNaiveClassifier, self).__init__(labels, *args, **kwargs)
        self.noisy_checkpoint = noisy_checkpoint
        
    def on_fit_start(self) -> None:
        assert self.noisy_checkpoint is not None
        self.prev = copy.deepcopy(self.model).to(self.device)
        self.noisy_model = self.base_classifier.load_from_checkpoint(self.noisy_checkpoint).to(self.device)
        self.confusion_matrix = torch.eye(self.num_classes, requires_grad=False).to(self.device)
        self.identity = torch.eye(self.num_classes, requires_grad=False).to(self.device)
        return super().on_fit_start()

    def on_train_epoch_start(self) -> None:
        self.prev.load_state_dict(self.model.state_dict())
        self.prev.eval()
        self.noisy_model.eval()
        return super().on_train_epoch_start()

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
        loss = F.nll_loss(log_logits, label)
        self.log('train_loss', loss)
        return loss

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
        loss = F.nll_loss(log_logits, label)
        self.log('val_loss', loss, on_epoch=True)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)
        
    def on_validation_epoch_end(self):
        self.confusion_matrix = self.val_metrics.cnfs_mat.compute().detach().nan_to_num(0, 0, 0)
        super().on_validation_epoch_end

    def test_step(self, batch, batch_idx):
        data, corrected_label = batch
        output = self(data)
        logits = F.softmax(output, -1)
        loss = F.cross_entropy(output, corrected_label)
        self.log('test_loss', loss, on_epoch=True)
        self.test_metrics(logits, corrected_label)
        self.log_scalars(self.test_metrics)
