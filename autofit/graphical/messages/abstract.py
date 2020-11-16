from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from operator import and_
from typing import Optional, Tuple

import numpy as np


class AbstractMessage(ABC):
    log_base_measure: float

    _parameter_support: Optional[Tuple[Tuple[float, float], ...]] = None
    _support: Optional[Tuple[Tuple[float, float], ...]] = None

    @property
    @abstractmethod
    def natural_parameters(self):
        pass

    @abstractmethod
    def sample(self, n_samples):
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
    @abstractmethod
    def mean(self):
        pass

    @property
    @abstractmethod
    def variance(self):
        pass

    @property
    def scale(self):
        return self.variance ** 0.5

    def __init__(self, parameters, log_norm=0.):
        self.log_norm = log_norm
        self._broadcast = np.broadcast(*parameters)
        self.parameters = parameters

    def __iter__(self):
        return iter(self.parameters)

    @property
    def shape(self):
        return self._broadcast.shape

    @property
    def ndim(self):
        return self._broadcast.ndim

    @classmethod
    def from_natural_parameters(cls, parameters, **kwargs):
        args = cls.invert_natural_parameters(parameters)
        return cls(*args, **kwargs)

    @classmethod
    @abstractmethod
    def invert_sufficient_statistics(cls, sufficient_statistics):
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
            t = self.to_canonical_form(x)
            log_base = self.log_base_measure
            eta_t = np.multiply(eta, t).sum(0)  # TODO this can be made more efficient using tensordot
            return log_base + eta_t - self.log_partition
        elif np.shape(x)[1:] == self.shape:
            return np.array([self.logpdf(x_) for x_ in x])

        raise ValueError(
            f"shape of passed value {x.shape} does not match message shape {self.shape}")

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    @classmethod
    def project(cls, samples, log_weights):
        """Calculates the sufficient statistics of a set of samples
        and returns the distribution with the appropriate parameters
        that match the sufficient statistics
        """
        # if weights aren't passed then equally weight all samples

        # Numerically stable weighting for very small/large weights
        log_w_max = np.max(log_weights, axis=0, keepdims=True)
        w = np.exp(log_weights - log_w_max)
        norm = w.mean(0)
        log_norm = np.log(norm) + log_w_max[0]

        tx = cls.to_canonical_form(samples)
        w /= norm
        suff_stats = (tx * w[None, ...]).mean(1)

        assert np.isfinite(suff_stats).all()
        return cls.from_sufficient_statistics(suff_stats, log_norm=log_norm)

    @classmethod
    def from_mode(cls, mode, covariance):
        pass

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

    @staticmethod
    def _iter_dists(dists):
        for elem in dists:
            if isinstance(elem, AbstractMessage):
                yield elem
            elif np.isscalar(elem):
                yield elem
            else:
                for dist in elem:
                    yield dist

    def update_invalid(self, other: 'AbstractMessage') -> 'AbstractMessage':
        invalid = reduce(
            and_, (np.isfinite(p) for p in
                   chain(self.parameters, [self.check_support()])))
        if self.ndim:
            valid_parameters = (
                np.where(invalid, p, p_safe) for p, p_safe in zip(self, other))
        else:
            valid_parameters = self if invalid else other  # TODO: Fairly certain this would not work
        return type(self)(*valid_parameters, log_norm=self.log_norm)

    def check_support(self) -> np.ndarray:
        if self._parameter_support is not None:
            return reduce(
                and_,
                ((p >= support[0]) & (p <= support[1])
                 for p, support in zip(self.parameters, self._parameter_support)))
        elif self.ndim:
            return np.array(True, dtype=bool, ndmin=self.ndim)
        return np.array([True])

    @property
    def is_valid(self):
        return np.isfinite(
            self.natural_parameters
        ).all() and np.all(
            self.check_support()
        )

    @staticmethod
    def _get_mean_variance(mean: np.ndarray, covariance: float
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
