import os
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset

from bnn.constants import DATA_DIR


class VisionDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, train_test_split: Tuple[int, int], batch_size: int = 128):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.is_setup = False
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.is_setup:
            self.is_setup = True
            data = np.load(os.path.join(DATA_DIR, self.dataset_name, "data.npy"))
            labels = np.load(os.path.join(DATA_DIR, self.dataset_name, "labels.npy"))
            corrected_labels = np.load(os.path.join(DATA_DIR, self.dataset_name, "corrected_labels.npy"))
            train_data = data[:self.train_test_split[0]]
            train_labels = labels[:self.train_test_split[0]]
            train_corrected_labels = corrected_labels[:self.train_test_split[0]]
            test_data = data[self.train_test_split[0]:]
            test_labels = labels[self.train_test_split[0]:]
            test_corrected_labels = corrected_labels[self.train_test_split[0]:]
            training_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels).long(),
                                             torch.tensor(train_corrected_labels).long())
            val_len = int(len(training_dataset) / 10)
            self.train_dataset, self.val_dataset = random_split(training_dataset,
                                                                [len(training_dataset) - val_len, val_len])
            self.test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels).long(),
                                              torch.tensor(test_corrected_labels).long())
            assert len(self.test_dataset) == self.train_test_split[1], \
                f'train_test_split is incorrect, test set has length{len(self.test_dataset)}, ' \
                f'expected {self.train_test_split[1]}'

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


data = VisionDataModule("mnist", (60000, 10000))
