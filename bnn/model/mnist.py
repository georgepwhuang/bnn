from typing import Union, List

from torch import nn

from bnn.model.base import BaseClassifier


class MNISTClassifier(BaseClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], multilabel: bool = False):
        super().__init__(labels, multilabel)
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        )