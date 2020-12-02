from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain
from operator import and_
from typing import Optional, Tuple, Union, Iterator
from inspect import getfullargspec

import numpy as np

from autofit.mapper.variable import Variable


class AbstractMessage(ABC):
    log_base_measure: float
    _multivariate: bool = False
    _parameter_support: Optional[Tuple[Tuple[float, float], ...]] = None
    _support: Optional[Tuple[Tuple[float, float], ...]] = None

    @property
    @abstractmethod
    def natural_parameters(self):
        pass

    @abstractmethod
    def sample(self, n_samples: Optional[int] = None):
        pass

    @staticmethod
    @abstractmethod
    def invert_natural_parameters(natural_parameters: np.ndarray
    ) -> Tuple[np.ndarray,  ...]:
        pass

    @staticmethod
    @abstractmethod
    def to_canonical_form(x: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def log_partition(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def variance(self) -> np.ndarray:
        pass

    @property
    def scale(self) -> np.ndarray:
        return self.variance ** 0.5

    def __init__(self, parameters: Tuple[np.ndarray, ...], log_norm=0.):
        self.log_norm = log_norm
        self._broadcast = np.broadcast(*parameters)
        if self.shape:
            self.parameters = tuple(
                np.asanyarray(p) for p in parameters)
        else:
            self.parameters = tuple(parameters)


    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.parameters)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._broadcast.shape

    @property
    def ndim(self) -> int: 
        return self._broadcast.ndim

    @classmethod
    def from_natural_parameters(
            cls, 
            parameters: Tuple[np.ndarray, ...], 
            **kwargs
    ) -> "AbstractMessage":
        args = cls.invert_natural_parameters(parameters)
        return cls(*args, **kwargs)

    @classmethod
    @abstractmethod
    def invert_sufficient_statistics(cls, sufficient_statistics: np.ndarray
    ) -> Tuple[np.ndarray, ...]:
        pass

    @classmethod
    def from_sufficient_statistics(cls, suff_stats: np.ndarray, **kwargs
    ) -> "AbstractMessage":
        natural_params = cls.invert_sufficient_statistics(suff_stats)
        return cls.from_natural_parameters(natural_params, **kwargs)

    def sum_natural_parameters(self, *dists: "AbstractMessage"
    ) -> "AbstractMessage":
        """return the unnormalised result of multiplying the pdf
        of this distribution with another distribution of the same
        type
        """
        new_params = sum(
            (dist.natural_parameters for dist in self._iter_dists(dists) 
            if isinstance(dist, AbstractMessage)),
            self.natural_parameters)
        mul_dist = self.from_natural_parameters(new_params)
        return mul_dist

    def sub_natural_parameters(self, other: "AbstractMessage"
    ) -> "AbstractMessage":
        """return the unnormalised result of dividing the pdf
        of this distribution with another distribution of the same
        type"""
        log_norm = self.log_norm - other.log_norm
        new_params = (self.natural_parameters - other.natural_parameters)
        div_dist = self.from_natural_parameters(new_params, log_norm=log_norm)
        return div_dist

    _multiply = sum_natural_parameters
    _divide = sub_natural_parameters

    def __mul__(self, other: "AbstractMessage") -> "AbstractMessage":
        if np.isscalar(other):
            log_norm = self.log_norm + np.log(other)
            return type(self)(*self.parameters, log_norm=log_norm)
        else:
            return self._multiply(other)

    def __rmul__(self, other: "AbstractMessage") -> "AbstractMessage":
        return self * other

    def __truediv__(self, other: "AbstractMessage") -> "AbstractMessage":
        if np.isscalar(other):
            log_norm = self.log_norm - np.log(other)
            return type(self)(*self.parameters, log_norm=log_norm)
        else:
            return self._divide(other)

    def __pow__(self, other: "AbstractMessage") -> "AbstractMessage":
        natural = self.natural_parameters
        new_params = other * natural
        log_norm = other * self.log_norm
        return self.from_natural_parameters(new_params, log_norm=log_norm)

    def __str__(self) -> str:
        param_attrs = [
            (attr, np.asanyarray(getattr(self, attr)))
            for attr in getfullargspec(self.__init__).args[1:-1]]
        if self.shape:
            pad = max(len(attr) for attr, _ in param_attrs)
            attr_str = "    {:<%d}={}" % pad
            param_strs = ',\n'.join(
                attr_str.format(
                    attr, np.array2string(val, prefix=' '*(pad + 5)))
                for attr, val in param_attrs)
            return f"{type(self).__name__}(\n{param_strs})"
        else:
            param_strs = ', '.join(
                attr + '=' + 
                np.array2string(val, prefix=' '*(len(attr) + 1))
                for attr, val in param_attrs)
            return f"{type(self).__name__}({param_strs})"

    __repr__ = __str__

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        shape = np.shape(x)
        if shape:
            x = np.asanyarray(x)

        if shape == self.shape:
            eta = self.natural_parameters
            t = self.to_canonical_form(x)
            log_base = self.log_base_measure
            # TODO this can be made more efficient using tensordot
            eta_t = np.multiply(eta, t).sum(0)  
            return log_base + eta_t - self.log_partition
        elif shape[1:] == self.shape:
            eta = self.natural_parameters
            t = self.to_canonical_form(x)
            eta_t = np.multiply(eta[:, None, ...], t).sum(0)
            return self.log_base_measure + eta_t - self.log_partition

        raise ValueError(
            f"shape of passed value {shape} does not "
            f"match message shape {self.shape}")

    def numerical_logpdf_gradient(self, x: np.ndarray, eps: float=1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        shape = np.shape(x)
        if shape:
            x0 = np.array(x, dtype=np.float64)
            logl0 = self.logpdf(x0)
            if self._multivariate:
                grad_logl = np.empty(logl0.shape + x0.shape)
                sl = tuple(slice(None) for _ in range(logl0.ndim))
                with np.nditer(x0, flags=['multi_index'], op_flags=['readwrite']) as it:
                    for xv in it:
                        xv += eps
                        logl = self.logpdf(x0)
                        grad_logl[sl + it.multi_index] = (logl - logl0)/eps
                        xv -= eps
            else:
                l0 = logl0.sum()
                grad_logl = np.empty_like(x0)
                with np.nditer(x0, flags=['multi_index'], op_flags=['readwrite']) as it:
                    for xv in it:
                        xv += eps
                        logl = self.logpdf(x0).sum()
                        grad_logl[it.multi_index] = (logl - l0)/eps
                        xv -= eps
        else:
            logl0 = self.logpdf(x)
            grad_logl = (self.logpdf(x + eps) - logl0)/eps

        return logl0, grad_logl

    def numerical_logpdf_gradient_hessian(self, x: np.ndarray, eps: float=1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        shape = np.shape(x)
        if shape:
            x0 = np.array(x, dtype=np.float64)
            if self._multivariate:
                logl0, gradl0 = self.numerical_logpdf_gradient(x0)
                hess_logl = np.empty(gradl0.shape + x0.shape)
                sl = tuple(slice(None) for _ in range(gradl0.ndim))
                with np.nditer(x0, flags=['multi_index'], op_flags=['readwrite']) as it:
                    for xv in it:
                        xv += eps
                        _, gradl = self.numerical_logpdf_gradient(x0)
                        hess_logl[sl + it.multi_index] = (gradl - gradl0)/eps
                        xv -= eps
            else:
                logl0 = self.logpdf(x0)
                l0 = logl0.sum()
                grad_logl = np.empty_like(x0)
                hess_logl = np.empty_like(x0)
                with np.nditer(x0, flags=['multi_index'], op_flags=['readwrite']) as it:
                    for xv in it:
                        xv += eps
                        l1 = self.logpdf(x0).sum()
                        xv -= 2 * eps
                        l2 = self.logpdf(x0).sum()
                        g1 = (l1 - l0)/eps
                        g2 = (l0 - l2)/eps
                        grad_logl[it.multi_index] = g1
                        hess_logl[it.multi_index] = (g1 - g2)/eps
                        xv += eps

                gradl0 = grad_logl
        else:
            logl0 = self.logpdf(x)
            logl1 = self.logpdf(x + eps)
            logl2 = self.logpdf(x - eps)
            gradl0 = (logl1 - logl0)/eps
            gradl1 = (logl0 - logl2)/eps
            hess_logl = (gradl0 - gradl1)/eps

        return logl0, gradl0, hess_logl

    logpdf_gradient = numerical_logpdf_gradient
    logpdf_gradient_hessian = numerical_logpdf_gradient_hessian

    @classmethod
    def project(cls, samples: np.ndarray, log_weights: np.ndarray
    ) -> "AbstractMessage":
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
    def from_mode(cls, mode: np.ndarray, covariance: np.ndarray
    ) -> "AbstractMessage":
        pass

    def log_normalisation(self, *dists: Union["AbstractMessage", float]
    ) -> np.ndarray:
        """
        Calculates the log of the integral of the product of a
        set of distributions

        NOTE: ignores log normalisation
        """
        # Remove floats from messages passed
        dists = [
            dist for dist in self._iter_dists(dists) 
            if isinstance(dist, AbstractMessage)]

        # Calculate log product of message normalisation
        log_norm = self.log_base_measure - self.log_partition
        log_norm += sum(
            dist.log_base_measure - dist.log_partition
            for dist in dists)

        # Calculate log normalisation of product of messages
        prod_dist = self.sum_natural_parameters(*dists)
        log_norm -= prod_dist.log_base_measure - prod_dist.log_partition

        return log_norm

    @staticmethod
    def _iter_dists(dists) -> Iterator[Union["AbstractMessage", float]]:
        for elem in dists:
            if isinstance(elem, AbstractMessage):
                yield elem
            elif np.isscalar(elem):
                yield elem
            else:
                for dist in elem:
                    yield dist

    def update_invalid(self, other: 'AbstractMessage') -> 'AbstractMessage':
        valid = self.check_valid()
        if self.ndim:
            valid_parameters = (
                np.where(valid, p, p_safe) for p, p_safe in zip(self, other))
        else:
            # TODO: Fairly certain this would not work
            valid_parameters = self if valid else other  
        return type(self)(*valid_parameters, log_norm=self.log_norm)

    def check_support(self) -> np.ndarray:
        if self._parameter_support is not None:
            return reduce(
                and_,
                ((p >= support[0]) & (p <= support[1])
                 for p, support in 
                 zip(self.parameters, self._parameter_support)))
        elif self.ndim:
            return np.array(True, dtype=bool, ndmin=self.ndim)
        return np.array([True])

    def check_finite(self) -> np.ndarray:
        return np.isfinite(self.natural_parameters).all(0)

    def check_valid(self) -> np.ndarray:
        return self.check_finite() & self.check_support()

    @property
    def is_valid(self) -> bool:
        return np.all(self.check_finite()) and np.all(self.check_support())

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

    def __call__(
        self, 
        x: np.ndarray, 
        _variables: Optional[Tuple[str]]=('x')
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if _variables is None:
            return self.logpdf(x)
        else:
            if 'x' in _variables:
                loglike, g = self.logpdf_gradient(x)
                g = np.expand_dims(g, list(range(loglike.ndim)))
                return loglike, (g,)
            else:
                return self.logpdf(x), ()

    def as_factor(self, variable: "Variable", name: Optional[str]=None
    ) -> "FactorJacobian":
        from autofit.graphical import FactorJacobian
        if name is None:
            shape = self.shape
            clsname = type(self).__name__
            family = clsname[:-7] if clsname.endswith('Message') else clsname
            name = f"{family}Likelihood" + (str(shape) if shape else '')

        return FactorJacobian(self, x=variable, name=name, vectorised=True)
