from typing import Union, List

from torch import nn

from bnn.layers import BBBMixtureConv2d, BBBMixtureLinear
from bnn.model.template.bayesian import BayesianClassifier


class MixtureCIFARClassifier(BayesianClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], *args, **kwargs):
        super(MixtureCIFARClassifier, self).__init__(labels, beta, *args, **kwargs)
        self.model = nn.Sequential(
            BBBMixtureConv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBMixtureConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBMixtureConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Softplus(),
            BBBMixtureConv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBMixtureConv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            BBBMixtureConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBMixtureConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            BBBMixtureConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Softplus(),
            nn.Flatten(),
            BBBMixtureLinear(2048, 256),
            nn.Softplus(),
            nn.Dropout(0.2),
            BBBMixtureLinear(256, 256),
            nn.Softplus(),
            nn.Dropout(0.2),
            BBBMixtureLinear(256, self.num_classes)
        )
