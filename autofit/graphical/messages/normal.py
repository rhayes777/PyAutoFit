import numpy as np

from autofit.graphical.messages.abstract import AbstractMessage
from autofit.mapper.prior.prior import GaussianPrior


class NormalMessage(AbstractMessage):
    @property
    def log_partition(self):
        eta1, eta2 = self.natural_parameters
        return - eta1 ** 2 / 4 / eta2 - np.log(-2 * eta2) / 2

    log_base_measure = - 0.5 * np.log(2 * np.pi)
    _support = ((-np.inf, np.inf),)
    _parameter_support = ((-np.inf, np.inf), (0, np.inf))

    def __init__(
            self,
            mu=0.,
            sigma=1.,
            log_norm=0.
    ):
        self.mu = mu
        self.sigma = sigma
        super().__init__(
            (mu, sigma),
            log_norm=log_norm
        )

    @classmethod
    def from_prior(
            cls,
            prior
    ):
        return NormalMessage(
            mu=prior.mean,
            sigma=prior.sigma
        )

    def as_prior(self):
        return GaussianPrior(
            mean=self.mu,
            sigma=self.sigma
        )

    @property
    def natural_parameters(self):
        return self.calc_natural_parameters(
            self.mu,
            self.sigma
        )

    @staticmethod
    def calc_natural_parameters(mu, sigma):
        precision = sigma ** -2
        return np.array([mu * precision, - precision / 2])

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        eta1, eta2 = natural_parameters
        mu = - 0.5 * eta1 / eta2
        sigma = np.sqrt(- 0.5 / eta2)
        return mu, sigma

    @staticmethod
    def to_canonical_form(x):
        return np.array([x, x ** 2])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        m1, m2 = suff_stats
        sigma = np.sqrt(m2 - m1 ** 2)
        return cls.calc_natural_parameters(m1, sigma)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        return self.sigma ** 2

    def sample(self, n_samples):
        x = np.random.randn(n_samples, *self.shape)
        mu, sigma = self.parameters
        if self.shape:
            return x * sigma[None, ...] + mu[None, ...]

        return x * sigma + mu

    @classmethod
    def from_mode(cls, mode: np.ndarray, covariance: float = 1.):
        mode, variance = cls._get_mean_variance(mode, covariance)
        return cls(mode, variance ** 0.5)
