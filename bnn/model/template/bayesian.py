from typing import List, Union

from torch.nn import functional as F

from bnn.model.template.base import BaseClassifier


class BayesianClassifier(BaseClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], *args, **kwargs):
        super(BayesianClassifier, self).__init__(labels, *args, **kwargs)
        self.beta = beta

    def training_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        celoss = F.cross_entropy(output, label)
        self.log('train_cross_entropy_loss', celoss)
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
        celoss = F.cross_entropy(output, label)
        self.log('val_cross_entropy_loss', celoss,  on_epoch=True)
        klloss = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_loss'):
                klloss = klloss + module.kl_loss()
        self.log('val_kldiv_loss', klloss)
        total_loss = celoss + self.get_beta(batch_idx, len(self.trainer.val_dataloaders), self.current_epoch, self.trainer.max_epochs) * klloss
        self.log('val_total_loss', total_loss)
        logits = F.softmax(output, -1)
        self.val_metrics(logits, label)
        self.log_scalars(self.val_metrics)

    def on_validation_epoch_end(self):
        self.log_nonscalars(self.val_metrics)

    def test_step(self, batch, batch_idx):
        data, label, corrected_label = batch
        output = self(data)
        celoss = F.cross_entropy(output, label)
        self.log('test_cross_entropy_loss', celoss, on_epoch=True)
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

    def on_test_epoch_end(self):
        self.log_nonscalars(self.test_metrics)

    def get_beta(self, batch_idx, m, epoch, num_epochs):
        beta_type = self.beta
        if type(beta_type) is float:
            return beta_type

        if beta_type == "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
        elif beta_type == "Soenderby":
            if epoch is None or num_epochs is None:
                raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
            beta = min(epoch / (num_epochs // 4), 1)
        elif beta_type == "Standard":
            beta = 1 / m
        else:
            beta = 0
        return beta