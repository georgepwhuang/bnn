from typing import Union, List

import torch

from bnn.model.cifar.cifar import CIFARClassifier
from bnn.model.template.shifted_naive import ShiftedNaiveClassifier


class ShiftedCIFARClassifier(ShiftedNaiveClassifier, CIFARClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], correction_matrix: torch.Tensor, *args, **kwargs):
        super(ShiftedCIFARClassifier, self).__init__(labels=labels, correction_matrix=correction_matrix, *args,
                                                     **kwargs)
