from typing import Union, List

from bnn.model.mnist.bayesian_mnist import BayesianMNISTClassifier
from bnn.model.template.esti_bayesian import EstiBayesianClassifier


class EstiBayesianMNISTClassifier(EstiBayesianClassifier, BayesianMNISTClassifier):
    def __init__(self, labels: Union[List[Union[str, int]], int], beta: Union[str, float], noisy_checkpoint: str, *args, **kwargs):
        super(EstiBayesianMNISTClassifier, self).__init__(labels = labels, beta=beta, noisy_checkpoint = noisy_checkpoint, *args, **kwargs)
        self.base_classifier = BayesianMNISTClassifier
        