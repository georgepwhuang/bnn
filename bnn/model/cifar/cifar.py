from typing import Union, List

from torch import nn

from bnn.model.template.base import BaseClassifier


class CIFARClassifier(BaseClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], *args, **kwargs):
        super(CIFARClassifier, self).__init__(labels, *args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Softplus(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        