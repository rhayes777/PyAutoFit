import math
from typing import Tuple, Union

import numpy as np
from autofit.jax_wrapper import erfinv
from scipy.stats import norm

from autoconf import cached_property
from autofit.mapper.operator import LinearOperator
from autofit.messages.abstract import AbstractMessage
from .composed_transform import TransformedMessage
from .transform import (
    phi_transform,
    log_transform,
    multinomial_logit_transform,
    log_10_transform,
)
from .. import exc


def is_nan(value):
    is_nan_ = np.isnan(value)
    if isinstance(is_nan_, np.ndarray):
        is_nan_ = is_nan_.all()
    return is_nan_


class NormalMessage(AbstractMessage):
    @cached_property
    def log_partition(self):
        eta1, eta2 = self.natural_parameters
        return -(eta1**2) / 4 / eta2 - np.log(-2 * eta2) / 2

    log_base_measure = -0.5 * np.log(2 * np.pi)
    _support = ((-np.inf, np.inf),)
    _parameter_support = ((-np.inf, np.inf), (0, np.inf))

    def __init__(
        self,
        mean,
        sigma,
        lower_limit=-math.inf,
        upper_limit=math.inf,
        log_norm=0.0,
        id_=None,
    ):
        if (np.array(sigma) < 0).any():
            raise exc.MessageException("Sigma cannot be negative")

        super().__init__(
            mean,
            sigma,
            log_norm=log_norm,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
        )
        self.mean, self.sigma = self.parameters

    def cdf(self, x):
        return norm.cdf(x, loc=self.mean, scale=self.sigma)

    def ppf(self, x):
        return norm.ppf(x, loc=self.mean, scale=self.sigma)

    @cached_property
    def natural_parameters(self):
        return self.calc_natural_parameters(self.mean, self.sigma)

    @staticmethod
    def calc_natural_parameters(mu, sigma):
        precision = 1 / sigma**2
        return np.array([mu * precision, -precision / 2])

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        eta1, eta2 = natural_parameters
        mu = -0.5 * eta1 / eta2
        sigma = np.sqrt(-0.5 / eta2)
        return mu, sigma

    @staticmethod
    def to_canonical_form(x):
        return np.array([x, x**2])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        m1, m2 = suff_stats
        sigma = np.sqrt(m2 - m1**2)
        return cls.calc_natural_parameters(m1, sigma)

    @cached_property
    def variance(self):
        return self.sigma**2

    def sample(self, n_samples=None):
        if n_samples:
            x = np.random.randn(n_samples, *self.shape)
            if self.shape:
                return x * self.sigma[None, ...] + self.mean[None, ...]
        else:
            x = np.random.randn(*self.shape)

        return x * self.sigma + self.mean

    def kl(self, dist):
        return (
            np.log(dist.sigma / self.sigma)
            + (self.sigma**2 + (self.mean - dist.mean) ** 2) / 2 / dist.sigma**2
            - 1 / 2
        )

    @classmethod
    def from_mode(
        cls, mode: np.ndarray, covariance: Union[float, LinearOperator] = 1.0, **kwargs
    ):
        if isinstance(covariance, LinearOperator):
            variance = covariance.diagonal()
        else:
            mode, variance = cls._get_mean_variance(mode, covariance)
        return cls(mode, np.abs(variance) ** 0.5, **kwargs)

    def _normal_gradient_hessian(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # raise Exception
        shape = np.shape(x)
        if shape:
            x = np.asanyarray(x)
            deltax = x - self.mean
            hess_logl = -self.sigma**-2
            grad_logl = deltax * hess_logl
            eta_t = 0.5 * grad_logl * deltax
            logl = self.log_base_measure + eta_t - np.log(self.sigma)

            if shape[1:] == self.shape:
                hess_logl = np.repeat(
                    np.reshape(hess_logl, (1,) + np.shape(hess_logl)), shape[0], 0
                )

        else:
            deltax = x - self.mean
            hess_logl = -self.sigma**-2
            grad_logl = deltax * hess_logl
            eta_t = 0.5 * grad_logl * deltax
            logl = self.log_base_measure + eta_t - np.log(self.sigma)

        return logl, grad_logl, hess_logl

    def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._normal_gradient_hessian(x)[:2]

    def logpdf_gradient_hessian(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._normal_gradient_hessian(x)

    __name__ = "gaussian_prior"

    __default_fields__ = ("log_norm", "id_")

    def value_for(self, unit: float) -> float:
        """
        Returns a physical value from an input unit value according to the Gaussian distribution of the prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.

        Returns
        -------
        value
            The unit value mapped to a physical value according to the prior.

        Examples
        --------

        prior = af.GaussianPrior(mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.5)
        """
        try:
            inv = erfinv(1 - 2.0 * (1.0 - unit))
        except TypeError:
            from scipy.special import erfinv as scipy_erfinv
            inv = scipy_erfinv(1 - 2.0 * (1.0 - unit))
        return self.mean + (self.sigma * np.sqrt(2) * inv)

    def log_prior_from_value(self, value):
        """
        Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.
        This is used by certain non-linear searches (e.g. Emcee) in the log likelihood function evaluation.
        Parameters
        ----------
        value
            The physical value of this prior's corresponding parameter in a `NonLinearSearch` sample.
        """
        return (value - self.mean) ** 2.0 / (2 * self.sigma**2.0)

    def __str__(self):
        """
        The line of text describing this prior for the model_mapper.info file
        """
        return f"NormalMessage, mean = {self.mean}, sigma = {self.sigma}"

    def __repr__(self):
        return (
            "<NormalMessage id={} mean={} sigma={} "
            "lower_limit={} upper_limit={}>".format(
                self.id, self.mean, self.sigma, self.lower_limit, self.upper_limit
            )
        )

    @property
    def natural(self):
        return NaturalNormal.from_natural_parameters(
            self.natural_parameters * 0.0, **self._init_kwargs
        )

    def zeros_like(self) -> "AbstractMessage":
        return self.natural.zeros_like()


class NaturalNormal(NormalMessage):
    """
    Identical to the NormalMessage but allows non-normalised values,
    e.g negative or infinite variances
    """

    _parameter_support = ((-np.inf, np.inf), (-np.inf, 0))

    def __init__(
        self,
        eta1,
        eta2,
        lower_limit=-math.inf,
        upper_limit=math.inf,
        log_norm=0.0,
        id_=None,
    ):
        AbstractMessage.__init__(
            self,
            eta1,
            eta2,
            log_norm=log_norm,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
        )

    @cached_property
    def sigma(self):
        precision = -2 * self.parameters[1]
        return precision**-0.5

    @cached_property
    def mean(self):
        return np.nan_to_num(-self.parameters[0] / self.parameters[1] / 2)

    @staticmethod
    def calc_natural_parameters(eta1, eta2):
        return np.array([eta1, eta2])

    @cached_property
    def natural_parameters(self):
        return self.calc_natural_parameters(*self.parameters)

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        m1, m2 = suff_stats
        precision = 1 / (m2 - m1**2)
        return cls.calc_natural_parameters(m1 * precision, -precision / 2)

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        return natural_parameters

    @classmethod
    def from_mode(
        cls, mode: np.ndarray, covariance: Union[float, LinearOperator] = 1.0, **kwargs
    ):
        if isinstance(covariance, LinearOperator):
            precision = covariance.inv().diagonal()
        else:
            mode, variance = cls._get_mean_variance(mode, covariance)
            precision = 1 / variance

        return cls(mode * precision, -precision / 2, **kwargs)

    zeros_like = AbstractMessage.zeros_like

    @property
    def natural(self):
        return self


UniformNormalMessage = TransformedMessage(NormalMessage(0, 1), phi_transform)

Log10UniformNormalMessage = TransformedMessage(UniformNormalMessage, log_10_transform)

LogNormalMessage = TransformedMessage(NormalMessage(0, 1), log_transform)
Log10NormalMessage = TransformedMessage(NormalMessage(0, 1), log_10_transform)

# Support is the simplex
MultiLogitNormalMessage = TransformedMessage(
    NormalMessage(0, 1), multinomial_logit_transform
)
