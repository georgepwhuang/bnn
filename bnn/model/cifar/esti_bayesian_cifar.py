from typing import Union, List

from bnn.model.cifar.bayesian_cifar import BayesianCIFARClassifier
from bnn.model.template.esti_bayesian import EstiBayesianClassifier


class EstiBayesianCIFARClassifier(EstiBayesianClassifier, BayesianCIFARClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], noisy_checkpoint: str, *args, **kwargs):
        super(EstiBayesianCIFARClassifier, self).__init__(labels = labels, beta=beta, noisy_checkpoint = noisy_checkpoint, *args, **kwargs)
        self.base_classifier = BayesianCIFARClassifier
        