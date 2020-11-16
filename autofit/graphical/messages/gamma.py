import numpy as np
from scipy import special

from autofit.graphical.messages.abstract import AbstractMessage
from autofit.graphical.utils import invpsilog


class GammaMessage(AbstractMessage):
    @property
    def log_partition(self):
        alpha, beta = GammaMessage.invert_natural_parameters(
            self.natural_parameters
        )
        return special.gammaln(alpha) - alpha * np.log(beta)

    log_base_measure = 0.
    _support = ((0, np.inf),)
    _parameter_support = ((0, np.inf), (0, np.inf))

    def __init__(
            self,
            alpha=1.,
            beta=1.,
            log_norm=0.
    ):
        self.alpha = alpha
        self.beta = beta
        super().__init__(
            parameters=[
                alpha, beta
            ],
            log_norm=log_norm
        )

    @property
    def natural_parameters(self):
        return self.calc_natural_parameters(
            self.alpha,
            self.beta
        )

    @staticmethod
    def calc_natural_parameters(alpha, beta):
        return np.array([alpha - 1, - beta])

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

    @property
    def mean(self):
        return self.alpha / self.beta

    @property
    def variance(self):
        return self.alpha / self.beta ** 2

    def sample(self, n_samples):
        a1, b1 = self.parameters
        return np.random.gamma(a1, scale=1 / b1, size=(n_samples,) + self.shape)

    @classmethod
    def from_mode(cls, mode, covariance):
        m, V = cls._get_mean_variance(mode, covariance)

        alpha = 1 + m ** 2 * V  # match variance
        beta = alpha / m  # match mean
        return cls(alpha, beta)