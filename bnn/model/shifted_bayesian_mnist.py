from typing import Union, List

from torch import nn

from bnn.layers import BBBConv2d, BBBLinear

from bnn.model.template.shifted_bayesian import ShiftedBayesianClassifier
from bnn.model.bayesian_mnist import BayesianMNISTClassifier


class ShiftedBayesianMNISTClassifier(ShiftedBayesianClassifier, BayesianMNISTClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], dataset_name: str, beta: Union[str, float], *args, **kwargs):
        super(ShiftedBayesianMNISTClassifier, self).__init__(labels=labels, dataset_name=dataset_name, beta=beta, *args, **kwargs)
