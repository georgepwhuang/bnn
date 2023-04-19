from typing import Union, List

from bnn.model.mnist.mnist import MNISTClassifier
from bnn.model.template.esti_naive import EstiNaiveClassifier


class EstiMNISTClassifier(EstiNaiveClassifier, MNISTClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], noisy_checkpoint: str, *args, **kwargs):
        super(EstiMNISTClassifier, self).__init__(labels = labels, noisy_checkpoint = noisy_checkpoint, *args, **kwargs)
        self.base_classifier = MNISTClassifier
        