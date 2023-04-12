from typing import Union, List

from torch import nn

from bnn.layers import BBBMixtureConv2d, BBBMixtureLinear
from bnn.model.template.bayesian import BayesianClassifier


class MixtureMNISTClassifier(BayesianClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], *args, **kwargs):
        super(MixtureMNISTClassifier, self).__init__(labels, beta, *args, **kwargs)
        self.model = nn.Sequential(
            BBBMixtureConv2d(1, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2),
            BBBMixtureConv2d(8, 4, kernel_size=3),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            BBBMixtureLinear(100, 64),
            nn.Softplus(),
            nn.Dropout(0.2),
            BBBMixtureLinear(64, self.num_classes)
        )
