from typing import Union, List

from torch import nn

from bnn.model.shifted_naive import ShiftedNaiveClassifier
from bnn.model.mnist import MNISTClassifier


class ShiftedMNISTClassifier(ShiftedNaiveClassifier, MNISTClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], dataset_name: str, multilabel: bool = False):
        super(ShiftedMNISTClassifier, self).__init__(labels=labels, dataset_name=dataset_name, multilabel=multilabel)