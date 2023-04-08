from typing import Union, List

from torch import nn

from bnn.layers import BBBConv2d, BBBLinear
from bnn.model.template.bayesian import BayesianClassifier


class BayesianMNISTClassifier(BayesianClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], *args, **kwargs):
        super(BayesianMNISTClassifier, self).__init__(labels, beta, *args, **kwargs)
        self.model = nn.Sequential(
            BBBConv2d(1, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2),
            BBBConv2d(8, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            BBBLinear(100, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            BBBLinear(64, self.num_classes)
        )
