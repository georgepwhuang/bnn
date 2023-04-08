import json
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import random_split, DataLoader

from bnn.constants import DATA_DIR


class VisionDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, batch_size: int = 128):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.is_setup = False
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.is_setup:
            self.is_setup = True
            try:
                dataset_class = getattr(torchvision.datasets, self.dataset_name.upper())
            except AttributeError:
                raise ModuleNotFoundError(f"Dataset {self.dataset_name.upper()} not supplied in torchvision")
            self.full_dataset = dataset_class(root=os.path.join(DATA_DIR, "data", self.dataset_name.lower()),
                                              train=True, download=True, transform=torchvision.transforms.ToTensor())
            self.test_dataset = dataset_class(root=os.path.join(DATA_DIR, "data", self.dataset_name.lower()),
                                              train=False, download=True, transform=torchvision.transforms.ToTensor())
            self.num_classes = len(self.test_dataset.classes)
            val_len = int(len(self.full_dataset) / 10)
            self.train_dataset, self.val_dataset = random_split(self.full_dataset,
                                                                [len(self.full_dataset) - val_len, val_len])
            correction_matrix = np.zeros((self.num_classes, self.num_classes))
            for i in self.test_dataset.targets:
                correction_matrix[i][i] += 1
            with open(os.path.join(DATA_DIR, "corrections", self.dataset_name + ".json"), 'r') as f:
                corrections = json.load(f)
                for correction in corrections:
                    if correction['mturk']['guessed'] >= 3:
                        self.test_dataset.targets[correction['id']] = correction['our_guessed_label']
                        correction_matrix[correction['given_original_label']][correction['given_original_label']] -= 1
                        correction_matrix[correction['our_guessed_label']][correction['given_original_label']] += 1
            correction_matrix = torch.tensor(correction_matrix).float()
            correction_matrix = correction_matrix / correction_matrix.sum(dim=1)
            self.correction_matrix = correction_matrix.transpose(0, 1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
