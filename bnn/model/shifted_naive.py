import os
from typing import List, Union

import numpy as np
import torch
from torch.nn import functional as F

from bnn.constants import DATA_DIR
from bnn.metrics import MulticlassClassificationMetrics, BinaryClassificationMetrics, MultilabelClassificationMetrics
from bnn.model.base import BaseClassifier


class ShiftedNaiveClassifier(BaseClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], dataset_name: str, multilabel: bool = False):
        super().__init__(labels, multilabel)
        self.shifter_matrix = torch.tensor(
            np.load(os.path.join(DATA_DIR, dataset_name, "correction_matrix.npy"))).float()

        if self.task == "multiclass":
            self.val_true_metrics = MulticlassClassificationMetrics(self.num_classes, "val_true", self.labels)
            self.test_true_metrics = MulticlassClassificationMetrics(self.num_classes, "test_true", self.labels)
        elif self.task == "binary":
            self.val_true_metrics = BinaryClassificationMetrics("val_true")
            self.test_true_metrics = BinaryClassificationMetrics("test_true")
        elif self.task == "multilabel":
            self.val_true_metrics = MultilabelClassificationMetrics(self.num_classes, "val_true", self.labels)
            self.test_true_metrics = MultilabelClassificationMetrics(self.num_classes, "test_true", self.labels)

    def training_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        result = self(data)
        output = F.linear(result, self.shifter_matrix)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.float())
            with torch.no_grad():
                true_loss = F.binary_cross_entropy_with_logits(result, corrected_label.float())
        elif self.task == "multiclass":
            loss = F.cross_entropy(output, label)
            with torch.no_grad():
                true_loss = F.cross_entropy(result, corrected_label)
        elif self.task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(output, label.float())
            with torch.no_grad():
                true_loss = F.binary_cross_entropy_with_logits(result, corrected_label.float())
        else:
            raise RuntimeError(f"No task {self.task} is defined")
        self.log('train_loss', loss)
        self.log('true_train_loss', true_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        result = self(data)
        output = F.linear(result, self.shifter_matrix)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.float())
            logits = torch.sigmoid(output)
            with torch.no_grad():
                result = result.squeeze(1)
                true_loss = F.binary_cross_entropy_with_logits(result, corrected_label.float())
                true_logits = torch.sigmoid(result)
        elif self.task == "multiclass":
            loss = F.cross_entropy(output, label)
            out = F.softmax(output, -1)
            logits = out.argmax(dim=1)
            with torch.no_grad():
                true_loss = F.cross_entropy(result, corrected_label)
                true_out = F.softmax(output, -1)
                true_logits = true_out.argmax(dim=1)
        elif self.task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(output, label.float())
            logits = torch.sigmoid(output)
            with torch.no_grad():
                true_loss = F.binary_cross_entropy_with_logits(result, corrected_label.float())
                true_logits = torch.sigmoid(result)
        else:
            raise RuntimeError(f"No task {self.task} is defined")
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_true_loss', true_loss, on_epoch=True)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)
        self.val_true_metrics(true_logits, corrected_label)
        self.log_scalars(self.val_true_metrics)

    def test_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        result = self(data)
        output = F.linear(result, self.shifter_matrix)
        if self.task == "binary":
            output = output.squeeze(1)
            loss = F.binary_cross_entropy_with_logits(output, label.float())
            logits = torch.sigmoid(output)
            with torch.no_grad():
                result = result.squeeze(1)
                true_loss = F.binary_cross_entropy_with_logits(result, corrected_label.float())
                true_logits = torch.sigmoid(result)
        elif self.task == "multiclass":
            loss = F.cross_entropy(output, label)
            out = F.softmax(output, -1)
            logits = out.argmax(dim=1)
            with torch.no_grad():
                true_loss = F.cross_entropy(result, corrected_label)
                true_out = F.softmax(output, -1)
                true_logits = true_out.argmax(dim=1)
        elif self.task == "multilabel":
            loss = F.binary_cross_entropy_with_logits(output, label.float())
            logits = torch.sigmoid(output)
            with torch.no_grad():
                true_loss = F.binary_cross_entropy_with_logits(result, corrected_label.float())
                true_logits = torch.sigmoid(result)
        else:
            raise RuntimeError(f"No task {self.task} is defined")
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_true_loss', true_loss, on_epoch=True)
        self.test_metrics(logits, label)
        self.log_scalars(self.test_metrics)
        self.test_true_metrics(true_logits, corrected_label)
        self.log_scalars(self.test_true_metrics)
