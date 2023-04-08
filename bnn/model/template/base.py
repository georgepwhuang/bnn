from abc import ABC
from typing import List, Union

import pytorch_lightning as pl
from torch.nn import functional as F

from bnn.metrics import MulticlassClassificationMetrics


class BaseClassifier(pl.LightningModule, ABC):
    def __init__(self, labels: Union[List[Union[str, int]], int], *args, **kwargs):
        super().__init__()
        if isinstance(labels, int):
            self.num_classes = labels
            self.labels = list(map(lambda x: str(x), range(labels)))
        elif isinstance(labels, list):
            self.labels = list(map(lambda x: str(x), labels))
            self.num_classes = len(self.labels)
        self.save_hyperparameters()

        self.val_metrics = MulticlassClassificationMetrics(self.num_classes, "val", self.labels)
        self.test_metrics = MulticlassClassificationMetrics(self.num_classes, "test", self.labels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = F.cross_entropy(output, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = F.cross_entropy(output, label)
        logits = F.softmax(output, -1)
        self.log('val_loss', loss, on_epoch=True)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)

    def on_validation_epoch_end(self):
        self.log_nonscalars(self.val_metrics)

    def test_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = F.cross_entropy(output, label)
        logits = F.softmax(output, -1)
        self.log('test_loss', loss, on_epoch=True)
        self.test_metrics(logits, label)
        self.log_scalars(self.test_metrics)

    def on_test_epoch_end(self):
        self.log_nonscalars(self.test_metrics)

    def log_scalars(self, metric):
        for k, v in metric.scalars.items():
            self.log(f"{metric.mode}_{k}", v, on_epoch=True, prog_bar=True)

    def log_nonscalars(self, metric):
        for m_out in metric.nonscalars(self.current_epoch):
            if m_out["type"] == "fig":
                self.logger.experiment.add_figure(f"{metric.mode}_{m_out['name']}",
                                                  m_out['data'], global_step=self.current_epoch)
            elif m_out["type"] == "text":
                self.logger.experiment.add_text(f"{metric.mode}_{m_out['name']}",
                                                m_out['data'], global_step=self.current_epoch)
