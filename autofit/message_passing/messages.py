from abc import ABC, abstractmethod
from collections import namedtuple
from functools import reduce
from itertools import chain
from operator import and_
from typing import (
    NamedTuple, Dict, Tuple, Optional, Union, Iterator
)

import numpy as np
from scipy import stats, special
from scipy.special import logsumexp

from autofit.message_passing.utils import invpsilog

MAX_REPR_SIZE = 10


def _round(params, decimals=1):
    return type(params)(*(np.round(x, decimals) for x in params))


class NormalParams(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray

    @property
    def dist(self):
        """return scipy normal dist of the parameters"""
        return stats.norm(self.mu, self.sigma)

    round = _round


NormalParams.__doc__ = """NormalParams(mu, sigma)

Standard parameters for a normal distribution with pdf

f_N(x) = 1/sqrt(2 * pi * sigma**2) * exp(- (x - mu)**2 / (2 * sigma**2))

conditions
----------
mu is real
sigma > 0
"""


class GammaParams(NamedTuple):
    alpha: np.ndarray
    beta: np.ndarray

    @property
    def dist(self):
        """return scipy gamma dist of the parameters"""
        return stats.gamma(self.alpha, scale=1 / self.beta)

    round = _round


GammaParams.__doc__ = """GammaParams(alpha, beta)

Standard parameters for a gamma distribution with PDF:

f_G(x) = beta ** alpha / gamma(alpha) * x ** (alpha - 1) * exp(-beta * x)

conditions
----------
alpha > 0
beta > 0
"""


class NormalGammaParams(NamedTuple):
    alpha: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    lam: np.ndarray

    round = _round


NormalGammaParams.__doc__ = """NormalGammaParams(alpha, beta, mu, lam)

Standard parameters for a normal gamma distribution with PDF:

f_NG(x, t) = (
    beta ** alpha / gamma(alpha) * sqrt(lam/(2 * pi)) 
    * t ** (alpha - 1/2) * exp(-beta * t) * 
    exp( - lam * t * (x - mu)**2 / 2))

the PDF of the normal gamma can be expressed as the product of 
a Normal and gamma distribution, 

f_NG(x, t) = f_G(t; alpha, beta) * f_N(x, mu, 1/sqrt(lam * t))

conditions
----------
alpha > 0
beta > 0
mu is real
lam > 0
"""

DistributionParams = Union[
    Tuple, NormalParams, GammaParams, NormalGammaParams]


class AbstractMessage(ABC):
    _parameters = None
    _log_partition: Optional[np.ndarray] = None
    _sufficient_statistics: Optional[np.ndarray] = None
    _log_base_measure = NotImplemented
    _dist = None
    _fixed = False

    _Parameters: DistributionParams = staticmethod(lambda *x: tuple(x))
    _parameter_support: Optional[Tuple[Tuple[float, float], ...]] = None
    _support: Optional[Tuple[Tuple[float, float], ...]] = None
    _Projection = namedtuple(
        "Projection",
        ["sufficient_statistics", "variance", "effective_sample_size",
         "log_norm", "log_norm_var"])

    @staticmethod
    @abstractmethod
    def calc_natural_parameters(*parameters, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def invert_natural_parameters(natural_parameters):
        pass

    @staticmethod
    @abstractmethod
    def to_canonical_form(x):
        pass

    @staticmethod
    @abstractmethod
    def calc_log_partition(natural_parameters):
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

    def __init__(self, *args, log_norm=0., **kwargs):
        self.parameters = self._Parameters(*args, **kwargs)
        self.log_norm = log_norm
        self._broadcast = np.broadcast(*self.parameters)
        super().__init__()

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
    def from_sufficient_statistics(cls, suff_stats, **kwargs):
        natural_params = cls.invert_sufficient_statistics(suff_stats)
        return cls.from_natural_parameters(natural_params, **kwargs)

    @property
    def natural_parameters(self):
        if self._parameters is None:
            self._parameters = self.calc_natural_parameters(*self.parameters)
        return self._parameters

    @property
    def log_partition(self):
        if self._log_partition is None:
            self._log_partition = self.calc_log_partition(
                self.natural_parameters)
        return self._log_partition

    @property
    def sufficient_statistics(self):
        if self._sufficient_statistics is None:
            self._sufficient_statistics = self.calc_sufficient_statistics(
                self.natural_parameters)
        return self._sufficient_statistics

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

    def __getattr__(self, item):
        return getattr(
            self.parameters,
            item
        )

    def __repr__(self):
        params = self.parameters
        if self.size > MAX_REPR_SIZE:
            shapes = (", ".join(map(repr, np.shape(p))) for p in params)
            names = ((type(p).__name__) for p in params)
            params = self._Parameters(*(
                f"{name}[{shape}]" for name, shape in zip(names, shapes)))

            param_str = "(".join(str(params).split("(")[1:])
        else:
            param_str = "(".join(repr(params).split("(")[1:])
        return f"{type(self).__name__}({param_str}"

    def logpdf(self, x):
        if np.shape(x) == self.shape:
            eta = self.natural_parameters
            T = self.to_canonical_form(x)
            logbase = self.calc_log_base_measure(x)
            etaT = (eta * T).sum(0)  # TODO this can be made more efficient using tensordot
            return logbase + etaT - self.log_partition
        elif np.shape(x)[1:] == self.shape:
            return np.array([self.logpdf(x_) for x_ in x])
        else:
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

    def rvs(self, *args, **kwargs):
        return self.dist.rvs(*args, **kwargs)

    def sample(self, n_samples, *args, **kwargs):
        shape = self.shape
        return np.array(
            [self.rvs(*args, **kwargs).reshape(shape) for _ in range(n_samples)])

    @classmethod
    def default(cls) -> 'AbstractMessage':
        return cls(*cls._default_params)

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

    @property
    def is_fixed(self):
        return self._fixed

    @property
    def is_free(self):
        return ~ self.is_fixed

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
    _fixed = True

    def __init__(self, parameters, log_norm=0.):
        self.parameters = parameters,
        self.log_norm = log_norm
        self._broadcast = np.broadcast(*self.parameters)
        ABC.__init__(self)

    @staticmethod
    def calc_natural_parameters(args):
        return args

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        return natural_parameters,

    @staticmethod
    def to_canonical_form(x):
        return x

    @staticmethod
    def calc_log_partition(natural_parameters):
        return 0.

    @classmethod
    def calc_sufficient_statistics(cls, natural_parameters):
        return natural_parameters

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
        return self.parameters[0]

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
    _log_base_measure = - 0.5 * np.log(2 * np.pi)
    _Parameters = NormalParams
    _support = ((-np.inf, np.inf),)
    _parameter_support = ((-np.inf, np.inf), (0, np.inf))
    _default_params = NormalParams(0., 1.)

    @staticmethod
    def calc_natural_parameters(mu, sigma):
        precision = sigma ** -2
        return np.array([mu * precision, - precision / 2])

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        eta1, eta2 = natural_parameters
        mu = - 0.5 * eta1 / eta2
        sigma = np.sqrt(- 0.5 / eta2)
        return NormalMessage._Parameters(mu, sigma)

    @staticmethod
    def to_canonical_form(x):
        return np.array([x, x ** 2])

    @staticmethod
    def calc_log_partition(natural_parameters):
        eta1, eta2 = natural_parameters
        return (- eta1 ** 2 / 4 / eta2 - np.log(-2 * eta2) / 2)

    @classmethod
    def calc_sufficient_statistics(cls, natural_parameters):
        eta1, eta2 = natural_parameters
        T1 = -eta1 / 2 / eta2
        T2 = eta1 ** 2 / 4 / eta2 - 0.5 / eta2
        return np.array([T1, T2])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        m1, m2 = suff_stats
        sigma = np.sqrt(m2 - m1 ** 2)
        return cls.calc_natural_parameters(m1, sigma)

    @property
    def dist(self):
        if self._dist is None:
            self._dist = self.parameters.dist

        return self._dist

    @property
    def mean(self):
        return self.parameters.mu

    @property
    def variance(self):
        return self.parameters.sigma ** 2

    def sample(self, n_samples, *args, **kwargs):
        x = np.random.randn(n_samples, *self.shape)
        mu, sigma = self.parameters
        if self.shape:
            return (x * sigma[None, ...] + mu[None, ...])
        else:
            return (x * sigma + mu)

    @classmethod
    def from_mode(cls, mode: np.ndarray, covariance: np.ndarray = 1., **kwargs):
        mode, variance = cls._get_mean_variance(mode, covariance)
        return cls(mode, variance ** 0.5)


class GammaMessage(AbstractMessage):
    _log_base_measure = 0.
    _Parameters = GammaParams
    _support = ((0, np.inf),)
    _parameter_support = ((0, np.inf), (0, np.inf))
    _default_params = GammaParams(1., 1.)

    @staticmethod
    def calc_natural_parameters(alpha, beta):
        return np.array([alpha - 1, - beta])

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        eta1, eta2 = natural_parameters
        return GammaMessage._Parameters(eta1 + 1, -eta2)

    @staticmethod
    def to_canonical_form(x):
        return np.array([np.log(x), x])

    @staticmethod
    def calc_log_partition(natural_parameters):
        alpha, beta = GammaMessage.invert_natural_parameters(
            natural_parameters)
        return (special.gammaln(alpha) - alpha * np.log(beta))

    @classmethod
    def calc_sufficient_statistics(cls, natural_parameters):
        alpha, beta = cls.invert_natural_parameters(natural_parameters)
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
    def dist(self):
        return self.parameters.dist

    @property
    def mean(self):
        return self.parameters.alpha / self.parameters.beta

    @property
    def variance(self):
        return (self.parameters.alpha / self.parameters.beta ** 2)

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


class NormalGammaMessage(NormalMessage):
    _Parameters = NormalGammaParams
    _support = ((-np.inf, np.inf), (0, np.inf),)
    _parameter_support = ((0, np.inf), (0, np.inf), (-np.inf, np.inf), (0, np.inf))
    _HumanParameters = namedtuple(
        "NormalGamma", "mean, variance, n_mean, n_variance")
    _default_params = NormalGammaParams(1., 1., 0., 1.)

    @staticmethod
    def calc_natural_parameters(alpha, beta, mu, lam):
        return np.array([
            alpha - 0.5,
            -beta - lam * mu ** 2 / 2,
            lam * mu,
            -lam / 2])

    @staticmethod
    def invert_natural_parameters(natural_parameters):
        eta1, eta2, eta3, eta4 = natural_parameters
        return (eta1 + 0.5,
                -eta2 + eta3 ** 2 / 4 / eta4,
                -eta3 / 2 / eta4,
                -2 * eta4)

    @staticmethod
    def to_canonical_form(xt):
        x, t = xt
        return np.array([
            np.log(t), t, t * x, t * x ** 2])

    @classmethod
    def calc_sufficient_statistics(cls, natural_parameters):
        alpha, beta, mu, lam = cls.invert_natural_parameters(natural_parameters)
        eta1, eta2, eta3, eta4 = natural_parameters

        T1 = special.digamma(alpha) - np.log(beta)
        T2 = alpha / beta
        T3 = mu * alpha / beta
        T4 = 1 / lam + mu * T3
        return np.array([T1, T2, T3, T4])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats):
        logT, T, TX, TX2 = suff_stats

        mu = TX / T
        lam = (TX2 - mu * TX) ** -1
        alpha = invpsilog(logT - np.log(T))
        beta = alpha / T
        return cls.calc_natural_parameters(alpha, beta, mu, lam)

    @staticmethod
    def calc_log_partition(eta):
        a, b, mu, lam = NormalGammaMessage.invert_natural_parameters(eta)
        return (
                special.gammaln(a) - 0.5 * np.log(lam) - a * np.log(b))

    @abstractmethod
    def from_mode(self, mode, covariance, **kwargs):
        raise NotImplementedError

    @property
    def mean(self):
        alpha, beta, mu, lam = self.parameters
        return np.array([mu, np.sqrt(beta / alpha)])

    @property
    def variance(self):
        alpha, beta, mu, lam = self.parameters
        return np.array([beta / lam / (alpha - 1), alpha / beta ** 2])

    def sample(self, n_samples):
        alpha, beta, mu, lam = self.parameters
        gamma = stats.gamma(alpha, scale=1 / beta)
        t = [gamma.rvs() for _ in range(n_samples)]
        x = [stats.norm.rvs(loc=mu, scale=(lam * t_) ** -0.5)
             for t_ in t]
        return np.c_[x, t].T


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
