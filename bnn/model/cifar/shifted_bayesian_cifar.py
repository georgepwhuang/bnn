from typing import Union, List

import torch

from bnn.model.cifar.bayesian_cifar import BayesianCIFARClassifier
from bnn.model.template.shifted_bayesian import ShiftedBayesianClassifier


class ShiftedBayesianCIFARClassifier(ShiftedBayesianClassifier, BayesianCIFARClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], correction_matrix: torch.Tensor,
                 beta: Union[str, float], *args, **kwargs):
        super(ShiftedBayesianCIFARClassifier, self).__init__(labels=labels, correction_matrix=correction_matrix,
                                                             beta=beta, *args, **kwargs)
