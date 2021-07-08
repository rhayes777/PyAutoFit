from typing import Tuple

import numpy as np

from autofit.graphical.messages.abstract import AbstractMessage
from autofit.graphical.utils import cached_property
from autofit.mapper.prior.prior import GaussianPrior
from .transform import phi_transform, log_transform, multinomial_logit_transform


class NormalMessage(AbstractMessage):
    @cached_property
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
            log_norm=0.,
            id_=None,
            **kwargs
    ):
        super().__init__(
            mu, sigma,
            log_norm=log_norm,
            id_=id_
        )
        self.mu, self.sigma = self.parameters

    @classmethod
    def _from_prior(
            cls,
            prior: GaussianPrior
    ):
        return NormalMessage(
            mu=prior.mean,
            sigma=prior.sigma,
            id_=prior.id
        )

    def as_prior(self):
        return GaussianPrior(
            mean=self.mu,
            sigma=self.sigma
        )

    @cached_property
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

    @cached_property
    def mean(self):
        return self.mu

    @cached_property
    def variance(self):
        return self.sigma ** 2

    def sample(self, n_samples=None):
        mu, sigma = self.parameters
        if n_samples:
            x = np.random.randn(n_samples, *self.shape)
            if self.shape:
                return x * sigma[None, ...] + mu[None, ...]
        else:
            x = np.random.randn(*self.shape)

        return x * sigma + mu

    def kl(self, dist):
        return (
                np.log(dist.sigma / self.sigma)
                + (self.sigma ** 2 + (self.mu - dist.mu) ** 2) / 2 / dist.sigma ** 2
                - 1 / 2
        )

    @classmethod
    def from_mode(cls, mode: np.ndarray, covariance: float = 1., id_=None):
        mode, variance = cls._get_mean_variance(mode, covariance)
        return cls(mode, variance ** 0.5, id_=id_)

    def _logpdf_gradient_hessian(self, x: np.ndarray
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # raise Exception
        shape = np.shape(x)
        if shape:
            x = np.asanyarray(x)
            deltax = x - self.mu
            hess_logl = - self.sigma ** -2
            grad_logl = deltax * hess_logl
            eta_t = 0.5 * grad_logl * deltax
            logl = self.log_base_measure + eta_t - np.log(self.sigma)

            if shape[1:] == self.shape:
                hess_logl = np.repeat(
                    np.reshape(hess_logl, (1,) + np.shape(hess_logl)),
                    shape[0], 0)

        else:
            deltax = x - self.mu
            hess_logl = - self.sigma ** -2
            grad_logl = deltax * hess_logl
            eta_t = 0.5 * grad_logl * deltax
            logl = self.log_base_measure + eta_t - np.log(self.sigma)

        return logl, grad_logl, hess_logl

    logpdf_gradient_hessian = _logpdf_gradient_hessian

    def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._logpdf_gradient_hessian(x)[:2]


UniformNormalMessage = NormalMessage.transformed(
    phi_transform, 'UniformNormalMessage')
LogNormalMessage = NormalMessage.transformed(
    log_transform, 'LogNormalMessage')
# Support is the simplex
MultiLogitNormalMessage = NormalMessage.transformed(
    multinomial_logit_transform, 'MultiLogitNormalMessage', ((0, 1),))
