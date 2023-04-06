from typing import Union, List

import torch

from bnn.model.template.shifted_bayesian import ShiftedBayesianClassifier
from bnn.model.mnist.bayesian_mnist import BayesianMNISTClassifier


class ShiftedBayesianMNISTClassifier(ShiftedBayesianClassifier, BayesianMNISTClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], correction_matrix: torch.Tensor, beta: Union[str, float], *args, **kwargs):
        super(ShiftedBayesianMNISTClassifier, self).__init__(labels=labels, correction_matrix=correction_matrix, beta=beta, *args, **kwargs)
