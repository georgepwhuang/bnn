from typing import Union, List

import torch
from bnn.model.mnist.mnist import MNISTClassifier
from bnn.model.template.shifted_naive import ShiftedNaiveClassifier


class ShiftedMNISTClassifier(ShiftedNaiveClassifier, MNISTClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], correction_matrix: torch.Tensor, *args, **kwargs):
        super(ShiftedMNISTClassifier, self).__init__(labels=labels, correction_matrix=correction_matrix, *args, **kwargs)
