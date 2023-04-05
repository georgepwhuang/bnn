from typing import Union, List

from bnn.model.mnist import MNISTClassifier
from bnn.model.shifted_naive import ShiftedNaiveClassifier


class ShiftedMNISTClassifier(ShiftedNaiveClassifier, MNISTClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], dataset_name: str):
        super(ShiftedMNISTClassifier, self).__init__(labels=labels, dataset_name=dataset_name)
