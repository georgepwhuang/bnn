from typing import Union, List

from torch import nn

from bnn.layers import BBBConv2d, BBBLinear

from bnn.model.template.bayesian import BayesianClassifier


class BayesianCIFARClassifier(BayesianClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], *args, **kwargs):
        super(BayesianCIFARClassifier, self).__init__(labels, beta, *args, **kwargs)
        self.model = nn.Sequential(
            BBBConv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            BBBConv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBConv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            BBBConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BBBConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            BBBConv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            BBBLinear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            BBBLinear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            BBBLinear(256, self.num_classes)
        )
