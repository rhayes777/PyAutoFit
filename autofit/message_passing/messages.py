from abc import ABC, abstractmethod
from collections import namedtuple
from functools import reduce
from itertools import chain
from operator import and_
from typing import (
    Dict, Tuple, Optional, Iterator
)

import numpy as np
from scipy import special
from scipy.special import logsumexp

from autofit.message_passing.utils import invpsilog


class Roundable(tuple):
    def round(self, decimals=1):
        return tuple(np.round(x, decimals) for x in self)


class AbstractMessage(ABC):
    _log_base_measure = NotImplemented

    _parameter_support: Optional[Tuple[Tuple[float, float], ...]] = None
    _support: Optional[Tuple[Tuple[float, float], ...]] = None
    _Projection = namedtuple(
        "Projection",
        ["sufficient_statistics", "variance", "effective_sample_size",
         "log_norm", "log_norm_var"])

    @property
    @abstractmethod
    def natural_parameters(self):
        pass

    @staticmethod
    @abstractmethod
    def invert_natural_parameters(natural_parameters):
        pass

    @staticmethod
    @abstractmethod
    def to_canonical_form(x):
        pass

    @property
    @abstractmethod
    def log_partition(self):
        pass

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    @property
    def scale(self):
        return self.variance ** 0.5

    def __init__(self, parameters, log_norm=0.):
        self.log_norm = log_norm
        self._broadcast = np.broadcast(*parameters)
        self.parameters = Roundable(parameters)

    def __iter__(self):
        return iter(self.parameters)

    @property
    def shape(self):
        return self._broadcast.shape

    @property
    def size(self):
        return self._broadcast.size

    @property
    def ndim(self):
        return self._broadcast.ndim

    @property
    def log_base_measure(self):
        return self._log_base_measure

    def calc_log_base_measure(self, x):
        if callable(self.log_base_measure):
            return self.log_base_measure(x)
        else:
            return self.log_base_measure

    @classmethod
    def from_natural_parameters(cls, parameters, **kwargs):
        args = cls.invert_natural_parameters(parameters)
        return cls(*args, **kwargs)

    @classmethod
    @abstractmethod
    def invert_sufficient_statistics(cls, suff_stats):
        pass

    @property
    @abstractmethod
    def sufficient_statistics(self):
        pass

    @classmethod
    def from_sufficient_statistics(cls, suff_stats, **kwargs):
        natural_params = cls.invert_sufficient_statistics(suff_stats)
        return cls.from_natural_parameters(natural_params, **kwargs)

    def sum_natural_parameters(self, *dists):
        """return the unnormalised result of multiplying the pdf
        of this distribution with another distribution of the same
        type
        """
        log_norm = self.log_norm + sum(
            dist.log_norm for dist in self._iter_dists(dists))
        new_params = sum(
            (dist.natural_parameters for dist in self._iter_dists(dists)),
            self.natural_parameters)
        mul_dist = self.from_natural_parameters(new_params, log_norm=log_norm)
        return mul_dist

    def sub_natural_parameters(self, other):
        """return the unnormalised result of dividing the pdf
        of this distribution with another distribution of the same
        type"""
        log_norm = self.log_norm - other.log_norm
        new_params = (self.natural_parameters - other.natural_parameters)
        div_dist = self.from_natural_parameters(new_params, log_norm=log_norm)
        return div_dist

    _multiply = sum_natural_parameters
    _divide = sub_natural_parameters

    def __mul__(self, other):
        if np.isscalar(other):
            log_norm = self.log_norm + np.log(other)
            return type(self)(*self.parameters, log_norm=log_norm)
        else:
            return self._multiply(other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if np.isscalar(other):
            log_norm = self.log_norm - np.log(other)
            return type(self)(*self.parameters, log_norm=log_norm)
        else:
            return self._divide(other)

    def __pow__(self, other):
        natural = self.natural_parameters
        new_params = other * natural
        log_norm = other * self.log_norm
        return self.from_natural_parameters(new_params, log_norm=log_norm)

    def __repr__(self):
        return f"{type(self).__name__}({self}"

    def __str__(self):
        return ""

    def logpdf(self, x):
        if np.shape(x) == self.shape:
            eta = self.natural_parameters
            T = self.to_canonical_form(x)
            logbase = self.calc_log_base_measure(x)
            etaT = (eta * T).sum(0)  # TODO this can be made more efficient using tensordot
            return logbase + etaT - self.log_partition
        elif np.shape(x)[1:] == self.shape:
            return np.array([self.logpdf(x_) for x_ in x])

        raise ValueError(
            f"shape of passed value {x.shape} does not match message shape {self.shape}")

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    logpdfs = logpdf
    pdfs = pdf

    @classmethod
    def project(cls, samples, log_weights=None, **kwargs):
        """Calculates the sufficient statistics of a set of samples
        and returns the distribution with the appropriate parameters
        that match the sufficient statistics
        """
        # if weights aren't passed then equally weight all samples
        n_samples = len(samples)
        if log_weights is None:
            w = np.ones_like(samples)
            norm = 1.
            log_norm = 0.
        #             log_norm_var = 0.
        else:
            # Numerically stable weighting for very small/large weights
            # log_norm = logsumexp(log_weights, axis=0) - np.log(n_samples)
            log_w_max = np.max(log_weights, axis=0, keepdims=True)
            w = np.exp(log_weights - log_w_max)
            norm = w.mean(0)
            log_norm = np.log(norm) + log_w_max[0]
        #             log_norm_var = np.log(w.var())

        TX = cls.to_canonical_form(samples)
        w /= norm
        suff_stats = (TX * w[None, ...]).mean(1)

        assert np.isfinite(suff_stats).all()
        return cls.from_sufficient_statistics(suff_stats, log_norm=log_norm)

    @classmethod
    def from_mode(self, mode, covariance, **kwargs):
        pass

    @classmethod
    def project_with_statistics(cls, samples, log_weights=None, **kwargs):
        """
        Calculates the sufficient statistics of a set of samples, the
        variances and the effective sample size for each statistic.
        """
        # if weights aren't passed then equally weight all samples
        n_samples = len(samples)
        if log_weights is None:
            w = np.ones(len(samples))
            log_norm = 0.
            log_norm_var = 0.
        else:
            # log_norm = logsumexp(log_weights, axis=0) - np.log(n_samples)
            w = np.exp(log_weights - logsumexp(log_weights, axis=0))
            norm = w.mean(0)
            w /= norm
            log_norm = np.log(norm)
            log_norm_var = np.log(w.var())

        TX = cls.to_canonical_form(samples)
        TXw = TX * w[None, ...]
        suff_stats = TXw.mean(1)
        var = np.square(TXw - suff_stats[:, None, ...]).mean(1)
        n_eff = np.abs(TXw).sum(1) ** 2 / (TXw ** 2).sum(1)

        assert np.isfinite(suff_stats).all()
        assert np.isfinite(var).all()
        return cls._Projection(suff_stats, var, n_eff, log_norm, log_norm_var)

    def log_normalisation(self, *dists):
        """
        Calculates the log of the integral of the product of a
        set of distributions

        NOTE: ignores log normalisation
        """
        prod_dist = self.sum_natural_parameters(self, *self._iter_dists(dists))

        log_numerator = sum(
            dist.log_base_measure - dist.log_partition
            for dist in dists)
        log_denominator = prod_dist.log_base_measure - prod_dist.log_partition

        return log_numerator - log_denominator

    def logBF_isequal(self, other, prior):
        """
        Calculates the log Bayes factor that this distribution is equal
        another given a prior distribution of the same type,

        evidence same = \int d1(x) * d2(x) * prior(x) dx
        evidence different = \int d1(x) * prior(x) dx * \int d2(x) * prior(x) dx

        logBF = log(evidence same) - log(evidence different)

        NOTE: ignores log normalisation
        """
        d1, d2 = self, other

        log_d1 = d1.log_normalisation(prior)
        log_d2 = d2.log_normalisation(prior)
        log_d12 = d1.log_normalisation(d2, prior)

        return log_d12 - log_d1 - log_d2

    def _iter_dists(self, dists):
        for elem in dists:
            if isinstance(elem, AbstractMessage):
                yield elem
            elif np.isscalar(elem):
                yield elem
            else:
                for dist in elem:
                    yield dist

    def sample(self, n_samples, *args, **kwargs):
        shape = self.shape
        return np.array(
            [self.rvs(*args, **kwargs).reshape(shape) for _ in range(n_samples)])

    def update_invalid(self, other: 'AbstractMessage') -> 'AbstractMessage':
        invalid = reduce(
            and_, (np.isfinite(p) for p in
                   chain(self.parameters, [self.check_support()])))
        if self.ndim:
            valid_parameters = (
                np.where(invalid, p, p_safe) for p, p_safe in zip(self, other))
        else:
            valid_parameters = self if invalid else other
        return type(self)(*valid_parameters, log_norm=self.log_norm)

    def check_support(self) -> np.ndarray:
        if self._parameter_support is not None:
            return reduce(
                and_,
                ((p >= support[0]) & (p <= support[1])
                 for p, support in zip(self.parameters, self._parameter_support)))
        elif self.ndim:
            return np.array(True, dtype=bool, ndmin=self.ndim)
        else:
            return np.array([True])

    @property
    def is_valid(self):
        return (np.isfinite(self.natural_parameters).all() and
                np.all(self.check_support()))

    @staticmethod
    def _get_mean_variance(mean: np.ndarray, covariance: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
        mean, covariance = np.asanyarray(mean), np.asanyarray(covariance)

        if not covariance.shape:
            # If variance is float simply pass through
            variance = covariance * np.ones_like(mean)
            if not variance.shape:
                variance = variance.item()
        elif mean.shape == covariance.shape:
            variance = np.asanyarray(covariance)
        elif covariance.shape == mean.shape * 2:
            # If 2D covariance matrix passed get diagonal
            inds = tuple(np.indices(mean.shape))
            variance = np.asanyarray(covariance)[inds * 2]
        else:
            raise ValueError(
                f"shape of covariance {covariance.shape} is invalid "
                f"must be (), {mean.shape}, or {mean.shape * 2}")
        return mean, variance


class FixedMessage(AbstractMessage):
    _log_base_measure = 0

    def __init__(self, value, log_norm=0.):
        self._value = value
        super().__init__(
            (value,),
            log_norm=log_norm
        )

    @property
    def natural_parameters(self):
        return self.parameters

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        return natural_parameters,

    @staticmethod
    def to_canonical_form(x):
        return x

    @property
    def log_partition(self):
        return 0.

    @property
    def sufficient_statistics(self):
        return self.natural_parameters

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        return suff_stats

    def sample(self, n_samples, *args, **kwargs):
        """
        Rely on array broadcasting to get fixed values to
        calculate correctly
        """
        return np.array(self.parameters)

    def logpdf(self, x):
        return np.zeros_like(x)

    logpdfs = logpdf

    @property
    def mean(self):
        return self._value

    @property
    def variance(self):
        return np.zeros_like(self.mean)

    def _no_op(self, *other, **kwargs):
        """
        'no-op' operation

        In many operations fixed messages should just
        return themselves
        """
        return self

    project = _no_op
    from_mode = _no_op
    __pow__ = _no_op
    __mul__ = _no_op
    __div__ = _no_op
    default = _no_op
    _multiply = _no_op
    _divide = _no_op
    sum_natural_parameters = _no_op
    sub_natural_parameters = _no_op


class NormalMessage(AbstractMessage):
    @property
    def log_partition(self):
        eta1, eta2 = self.natural_parameters
        return - eta1 ** 2 / 4 / eta2 - np.log(-2 * eta2) / 2

    _log_base_measure = - 0.5 * np.log(2 * np.pi)
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

    @property
    def sufficient_statistics(self):
        eta1, eta2 = self.natural_parameters
        T1 = -eta1 / 2 / eta2
        T2 = eta1 ** 2 / 4 / eta2 - 0.5 / eta2
        return np.array([T1, T2])

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

    def sample(self, n_samples, *args, **kwargs):
        x = np.random.randn(n_samples, *self.shape)
        mu, sigma = self.parameters
        if self.shape:
            return x * sigma[None, ...] + mu[None, ...]

        return x * sigma + mu

    @classmethod
    def from_mode(cls, mode: np.ndarray, covariance: np.ndarray = 1., **kwargs):
        mode, variance = cls._get_mean_variance(mode, covariance)
        return cls(mode, variance ** 0.5)


class GammaMessage(AbstractMessage):
    @property
    def log_partition(self):
        alpha, beta = GammaMessage.invert_natural_parameters(
            self.natural_parameters
        )
        return special.gammaln(alpha) - alpha * np.log(beta)

    _log_base_measure = 0.
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

    @property
    def sufficient_statistics(self):
        alpha, beta = self.invert_natural_parameters(
            self.natural_parameters
        )
        logX = special.digamma(alpha) - np.log(beta)
        X = alpha / beta
        return np.array([logX, X])

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

    def __add__(self, other):
        a1, b1 = self.parameters
        a2, b2 = other.parameters
        ab = a1 / b1 + a2 / b2
        ab2 = a1 / b1 ** 2 + a2 / b2 ** 2
        return GammaMessage(ab ** 2 / ab2, ab / ab2)

    def sample(self, n_samples, *args, **kwargs):
        a1, b1 = self.parameters
        return np.random.gamma(a1, scale=1 / b1, size=(n_samples,) + self.shape)

    @classmethod
    def from_mode(cls, mode, covariance, **kwargs):
        m, V = cls._get_mean_variance(mode, covariance)

        alpha = 1 + m ** 2 * V  # match variance
        beta = alpha / m  # match mean
        return cls(alpha, beta)


def map_dists(dists: Dict[str, AbstractMessage],
              values: Dict[str, np.ndarray],
              _call: str = 'logpdf'
              ) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Calls a method (default: logpdf) for each Message in dists
    on the corresponding value in values
    """
    for v in dists.keys() & values.keys():
        dist = dists[v]
        if isinstance(dist, AbstractMessage):
            yield v, getattr(dist, _call)(values[v])

# Message = Union[AbstractMessage, AbstractMessageBeliefMixin]
