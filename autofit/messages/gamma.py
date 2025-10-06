import math

import numpy as np
from scipy import special

from autoconf import cached_property
from autofit.messages.abstract import AbstractMessage
from autofit.messages.utils import invpsilog


class GammaMessage(AbstractMessage):
    @property
    def log_partition(self):
        alpha, beta = GammaMessage.invert_natural_parameters(self.natural_parameters)
        return special.gammaln(alpha) - alpha * np.log(beta)

    log_base_measure = 0.0
    _support = ((0, np.inf),)
    _parameter_support = ((0, np.inf), (0, np.inf))

    def __init__(
            self,
            alpha=1.0,
            beta=1.0,
            lower_limit=-math.inf,
            upper_limit=math.inf,
            log_norm=0.0,
            id_=None
    ):
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            alpha,
            beta,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            log_norm=log_norm,
            id_=id_
        )

    def value_for(self, unit: float) -> float:
        raise NotImplemented()

    @cached_property
    def natural_parameters(self):
        return self.calc_natural_parameters(self.alpha, self.beta)

    @staticmethod
    def calc_natural_parameters(alpha, beta):
        return np.array([alpha - 1, -beta])

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        eta1, eta2 = natural_parameters
        return eta1 + 1, -eta2

    @staticmethod
    def to_canonical_form(x):
        return np.array([np.log(x), x])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        logX, X = suff_stats
        alpha = invpsilog(logX - np.log(X))
        beta = alpha / X
        return cls.calc_natural_parameters(alpha, beta)

    @cached_property
    def mean(self):
        return self.alpha / self.beta

    @cached_property
    def variance(self):
        return self.alpha / self.beta ** 2

    def sample(self, n_samples=None):
        a1, b1 = self.parameters
        shape = (n_samples,) + self.shape if n_samples else self.shape
        return np.random.gamma(a1, scale=1 / b1, size=shape)

    @classmethod
    def from_mode(cls, mode, covariance, **kwargs):
        m, V = cls._get_mean_variance(mode, covariance)

        alpha = 1 + m ** 2 * V  # match variance
        beta = alpha / m  # match mean
        return cls(alpha, beta, **kwargs)

    def kl(self, dist):
        P, Q = dist, self
        logP = np.log(P.alpha)
        # TODO check this is correct
        # https://arxiv.org/pdf/0911.4863.pdf
        return (
                (P.alpha - Q.alpha) * special.psi(P.alpha)
                - special.gammaln(P.alpha)
                + special.gammaln(Q.alpha)
                + Q.alpha * (np.log(P.beta / Q.beta))
                + P.alpha * (Q.beta / P.beta - 1)
        )

    def logpdf_gradient(self, x):
        logl = self.logpdf(x)
        eta1 = self.natural_parameters[0]
        gradl = eta1 / x - self.beta
        return logl, gradl

    def logpdf_gradient_hessian(self, x):
        logl = self.logpdf(x)
        eta1 = self.natural_parameters[0]
        gradl = eta1 / x
        hessl = -gradl / x
        gradl -= self.beta
        return logl, gradl, hessl
