from typing import Union, List

from bnn.model.cifar.cifar import CIFARClassifier
from bnn.model.template.esti_naive import EstiNaiveClassifier


class EstiCIFARClassifier(EstiNaiveClassifier, CIFARClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], noisy_checkpoint: str, *args, **kwargs):
        super(EstiCIFARClassifier, self).__init__(labels = labels, noisy_checkpoint = noisy_checkpoint, *args, **kwargs)
        self.base_classifier = CIFARClassifier
        