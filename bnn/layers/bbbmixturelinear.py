import torch
import math
from bnn.layers.bbblinear import BBBLinear

class BBBMixtureLinear(BBBLinear):

    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBMixtureLinear, self).__init__(in_features, out_features, bias, priors)
        if priors is None:
            priors = {
                'prior_mu': (0, 0),
                'prior_sigma': (1, 0.01),
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

    def kl_loss(self):
        W_posterior_log = (-torch.log(self.W_sigma)).sum()
        W_prior_log_tensor = torch.stack([(-math.log(prior_sigma) - ((self.W_mu - prior_mu) ** 2) / (2 * prior_sigma ** 2)).sum()
                       for prior_sigma, prior_mu in zip(self.prior_sigma, self.prior_mu)], dim=0)
        W_prior_log = torch.logsumexp(W_prior_log_tensor, 0) / len(self.prior_sigma)
        kl = W_posterior_log - W_prior_log
        if self.use_bias:
            bias_posterior_log = (-torch.log(self.bias_sigma)).sum()
            bias_prior_log_tensor = torch.stack([(-math.log(prior_sigma) - ((self.bias_mu - prior_mu) ** 2) / (2 * prior_sigma ** 2)).sum()
                            for prior_sigma, prior_mu in zip(self.prior_sigma, self.prior_mu)], dim=0)
            bias_prior_log = torch.logsumexp(bias_prior_log_tensor, 0) / len(self.prior_sigma)
            kl += bias_posterior_log - bias_prior_log
        return kl
