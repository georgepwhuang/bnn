from typing import List, Union

import torch
from torch.nn import functional as F

from bnn.model.template.base import BaseClassifier


class ShiftedNaiveClassifier(BaseClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], correction_matrix: torch.Tensor, *args, **kwargs):
        super(ShiftedNaiveClassifier, self).__init__(labels, *args, **kwargs)
        self.shifter_matrix = correction_matrix
        self.shifter_matrix.requires_grad = False
        
    def on_fit_start(self) -> None:
        self.shifter_matrix = self.shifter_matrix.to(self.device)
        return super().on_fit_start()

    def on_test_start(self) -> None:
        self.shifter_matrix = self.shifter_matrix.to(self.device)
        return super().on_test_start()

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix) + 1e-16
        log_logits = torch.log(logits)
        loss = F.nll_loss(log_logits, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix) + 1e-16
        log_logits = torch.log(logits)
        loss = F.nll_loss(log_logits, label)
        self.log('val_loss', loss, on_epoch=True)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)

    def test_step(self, batch, batch_idx):
        data, corrected_label = batch
        output = self(data)
        logits = F.softmax(output, -1)
        loss = F.cross_entropy(output, corrected_label)
        self.log('test_loss', loss, on_epoch=True)
        self.test_metrics(logits, corrected_label)
        self.log_scalars(self.test_metrics)
