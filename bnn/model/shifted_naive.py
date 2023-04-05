import os
from typing import List, Union

import numpy as np
import torch
from torch.nn import functional as F

from bnn.constants import DATA_DIR
from bnn.metrics import MulticlassClassificationMetrics
from bnn.model.base import BaseClassifier


class ShiftedNaiveClassifier(BaseClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], dataset_name: str):
        super().__init__(labels)
        self.shifter_matrix = torch.tensor(
            np.load(os.path.join(DATA_DIR, dataset_name, "correction_matrix.npy"))).float()

        self.val_true_metrics = MulticlassClassificationMetrics(self.num_classes, "val_true", self.labels)
        self.test_true_metrics = MulticlassClassificationMetrics(self.num_classes, "test_true", self.labels)

    def training_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        loss = F.nll_loss(log_logits, label)
        with torch.no_grad():
            true_loss = F.cross_entropy(output, corrected_label)
        self.log('train_loss', loss)
        self.log('true_train_loss', true_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        loss = F.nll_loss(log_logits, label)
        with torch.no_grad():
            true_loss = F.cross_entropy(output, corrected_label)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_true_loss', true_loss, on_epoch=True)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)
        self.val_true_metrics(true_logits, corrected_label)
        self.log_scalars(self.val_true_metrics)
        
    def on_validation_epoch_end(self):
        self.log_nonscalars(self.val_metrics)
        self.log_nonscalars(self.val_true_metrics)

    def test_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        true_logits = F.softmax(output, -1)
        logits = F.linear(true_logits, self.shifter_matrix)
        log_logits = torch.log(logits)
        loss = F.nll_loss(log_logits, label)
        with torch.no_grad():
            true_loss = F.cross_entropy(output, corrected_label)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_true_loss', true_loss, on_epoch=True)
        self.test_metrics(logits, label)
        self.log_scalars(self.test_metrics)
        self.test_true_metrics(true_logits, corrected_label)
        self.log_scalars(self.test_true_metrics)

    def on_test_epoch_end(self):
        self.log_nonscalars(self.test_metrics)
        self.log_nonscalars(self.test_true_metrics)