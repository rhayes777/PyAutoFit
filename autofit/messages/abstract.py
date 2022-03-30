from abc import ABC, abstractmethod
from copy import copy
from functools import reduce
from inspect import getfullargspec
from numbers import Real
from operator import and_
from typing import Dict, Tuple, Iterator
from typing import Optional, Union, Type, List

import numpy as np

from autoconf import cached_property
from autofit.mapper.prior.abstract import Prior
from .transform import AbstractDensityTransform, LinearShiftTransform
from ..mapper.variable import Variable

enforce_id_match = True


def update_array(arr1, ind, arr2):
    if np.shape(arr1):
        out = arr1.copy()
        out[ind] = arr2
        return out

    return arr2


class AbstractMessage(Prior, ABC):
    log_base_measure: float
    _Base_class: Optional[Type["AbstractMessage"]] = None
    _projection_class: Optional[Type["AbstractMessage"]] = None
    _multivariate: bool = False
    _parameter_support: Optional[Tuple[Tuple[float, float], ...]] = None
    _support: Optional[Tuple[Tuple[float, float], ...]] = None

    def __init__(self, *parameters: Union[np.ndarray, float], log_norm=0.0, **kwargs):
        super().__init__(**kwargs)
        self.log_norm = log_norm
        self._broadcast = np.broadcast(*parameters)

        if self.shape:
            self.parameters = tuple(np.asanyarray(p) for p in parameters)
        else:
            self.parameters = tuple(parameters)

    def copy(self):
        cls = self._Base_class or type(self)
        result = cls(
            *(copy(params) for params in self.parameters),
            log_norm=self.log_norm,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
        )
        result.id = self.id
        return result

    def __bool__(self):
        return True

    @cached_property
    @abstractmethod
    def natural_parameters(self):
        pass

    @abstractmethod
    def sample(self, n_samples: Optional[int] = None):
        pass

    @staticmethod
    @abstractmethod
    def invert_natural_parameters(
            natural_parameters: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        pass

    @staticmethod
    @abstractmethod
    def to_canonical_form(x: np.ndarray) -> np.ndarray:
        pass

    @cached_property
    @abstractmethod
    def log_partition(self) -> np.ndarray:
        pass

    @cached_property
    @abstractmethod
    def variance(self) -> np.ndarray:
        pass

    @cached_property
    def scale(self) -> np.ndarray:
        return self.std

    @cached_property
    def std(self) -> np.ndarray:
        return self.variance ** 0.5

    def __hash__(self):
        return self.id

    @classmethod
    def calc_log_base_measure(cls, x):
        return cls.log_base_measure

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.parameters)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._broadcast.shape

    @property
    def size(self) -> int:
        return self._broadcast.size

    @property
    def ndim(self) -> int:
        return self._broadcast.ndim

    @classmethod
    def _cached_attrs(cls):
        for n in dir(cls):
            attr = getattr(cls, n)
            if isinstance(attr, cached_property):
                yield n

    def _reset_cache(self):
        for attr in self._cached_attrs():
            self.__dict__.pop(attr, None)

    def __getitem__(self, index) -> "AbstractMessage":
        cls = self._Base_class or type(self)
        if index == ():
            return self
        else:
            return cls(*(param[index] for param in self.parameters))

    def __setitem__(self, index, value):
        self._reset_cache()
        for param0, param1 in zip(self.parameters, value.parameters):
            param0[index] = param1

    def merge(self, index, value):
        cls = self._Base_class or type(self)
        return cls(
            *(
                update_array(param0, index, param1)
                for param0, param1 in zip(self.parameters, value.parameters)
            )
        )

    @classmethod
    def from_natural_parameters(
            cls, natural_parameters: np.ndarray, **kwargs
    ) -> "AbstractMessage":
        cls_ = cls._projection_class or cls._Base_class or cls
        args = cls_.invert_natural_parameters(natural_parameters)
        return cls_(*args, **kwargs)

    @classmethod
    @abstractmethod
    def invert_sufficient_statistics(
            cls, sufficient_statistics: np.ndarray
    ) -> np.ndarray:
        pass

    @classmethod
    def from_sufficient_statistics(
            cls, suff_stats: np.ndarray, **kwargs
    ) -> "AbstractMessage":
        natural_params = cls.invert_sufficient_statistics(suff_stats)
        cls_ = cls._projection_class or cls._Base_class or cls
        return cls_.from_natural_parameters(natural_params, **kwargs)

    def sum_natural_parameters(self, *dists: "AbstractMessage") -> "AbstractMessage":
        """return the unnormalised result of multiplying the pdf
        of this distribution with another distribution of the same
        type
        """
        new_params = sum(
            (
                dist.natural_parameters
                for dist in self._iter_dists(dists)
                if isinstance(dist, AbstractMessage)
            ),
            self.natural_parameters,
        )
        return self.from_natural_parameters(
            new_params,
            id_=self.id,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
        )

    def sub_natural_parameters(self, other: "AbstractMessage") -> "AbstractMessage":
        """return the unnormalised result of dividing the pdf
        of this distribution with another distribution of the same
        type"""
        log_norm = self.log_norm - other.log_norm
        new_params = self.natural_parameters - other.natural_parameters
        return self.from_natural_parameters(
            new_params,
            log_norm=log_norm,
            id_=self.id,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
        )

    _multiply = sum_natural_parameters
    _divide = sub_natural_parameters

    def __mul__(self, other: Union["AbstractMessage", Real]) -> "AbstractMessage":
        if isinstance(other, Prior):
            return self._multiply(other)
        else:
            cls = self._Base_class or type(self)
            log_norm = self.log_norm + np.log(other)
            return cls(
                *self.parameters,
                log_norm=log_norm,
                id_=self.id,
                lower_limit=self.lower_limit,
                upper_limit=self.upper_limit,
            )

    def __rmul__(self, other: "AbstractMessage") -> "AbstractMessage":
        return self * other

    def __truediv__(self, other: Union["AbstractMessage", Real]) -> "AbstractMessage":
        if isinstance(other, Prior):
            return self._divide(other)
        else:
            cls = self._Base_class or type(self)
            log_norm = self.log_norm - np.log(other)
            return cls(
                *self.parameters,
                log_norm=log_norm,
                id_=self.id,
                lower_limit=self.lower_limit,
                upper_limit=self.upper_limit,
            )

    def __pow__(self, other: Real) -> "AbstractMessage":
        natural = self.natural_parameters
        new_params = other * natural
        log_norm = other * self.log_norm
        new = self.from_natural_parameters(
            new_params,
            log_norm=log_norm,
            id_=self.id,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
        )
        return new

    @classmethod
    def parameter_names(cls):
        return getfullargspec(cls.__init__).args[1:-1]

    def __str__(self) -> str:
        param_attrs = [
            (attr, np.asanyarray(getattr(self, attr)))
            for attr in self.parameter_names()
        ]
        if self.shape:
            pad = max(len(attr) for attr, _ in param_attrs)
            attr_str = "    {:<%d}={}" % pad
            param_strs = ",\n".join(
                attr_str.format(attr, np.array2string(val, prefix=" " * (pad + 5)))
                for attr, val in param_attrs
            )
            return f"{type(self).__name__}(\n{param_strs})"
        else:
            param_strs = ", ".join(
                attr + "=" + np.array2string(val, prefix=" " * (len(attr) + 1))
                for attr, val in param_attrs
            )
            return f"{type(self).__name__}({param_strs})"

    __repr__ = __str__

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def _broadcast_natural_parameters(self, x):
        shape = np.shape(x)
        if shape == self.shape:
            return self.natural_parameters
        elif shape[1:] == self.shape:
            return self.natural_parameters[:, None, ...]
        else:
            raise ValueError(
                f"shape of passed value {shape} does not "
                f"match message shape {self.shape}"
            )

    def factor(self, x):
        # self.assert_within_limits(x)
        return self.logpdf(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        eta = self._broadcast_natural_parameters(x)
        t = self.to_canonical_form(x)
        log_base = self.calc_log_base_measure(x)
        return self.natural_logpdf(eta, t, log_base, self.log_partition)

    @classmethod
    def natural_logpdf(cls, eta, t, log_base, log_partition):
        eta_t = np.multiply(eta, t).sum(0)
        return np.nan_to_num(log_base + eta_t - log_partition, nan=-np.inf)

    def numerical_logpdf_gradient(
            self, x: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        shape = np.shape(x)
        if shape:
            x0 = np.array(x, dtype=np.float64)
            logl0 = self.logpdf(x0)
            if self._multivariate:
                grad_logl = np.empty(logl0.shape + x0.shape)
                sl = tuple(slice(None) for _ in range(logl0.ndim))
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        logl = self.logpdf(x0)
                        grad_logl[sl + it.multi_index] = (logl - logl0) / eps
                        xv -= eps
            else:
                l0 = logl0.sum()
                grad_logl = np.empty_like(x0)
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        logl = self.logpdf(x0).sum()  # type: ignore
                        grad_logl[it.multi_index] = (logl - l0) / eps
                        xv -= eps
        else:
            logl0 = self.logpdf(x)
            grad_logl = (self.logpdf(x + eps) - logl0) / eps

        return logl0, grad_logl

    def numerical_logpdf_gradient_hessian(
            self, x: np.ndarray, eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = np.shape(x)
        if shape:
            x0 = np.array(x, dtype=np.float64)
            if self._multivariate:
                logl0, gradl0 = self.numerical_logpdf_gradient(x0)
                hess_logl = np.empty(gradl0.shape + x0.shape)
                sl = tuple(slice(None) for _ in range(gradl0.ndim))
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        _, gradl = self.numerical_logpdf_gradient(x0)
                        hess_logl[sl + it.multi_index] = (gradl - gradl0) / eps
                        xv -= eps
            else:
                logl0 = self.logpdf(x0)
                l0 = logl0.sum()
                grad_logl = np.empty_like(x0)
                hess_logl = np.empty_like(x0)
                with np.nditer(x0, flags=["multi_index"], op_flags=["readwrite"]) as it:
                    for xv in it:
                        xv += eps
                        l1 = self.logpdf(x0).sum()
                        xv -= 2 * eps
                        l2 = self.logpdf(x0).sum()
                        g1 = (l1 - l0) / eps
                        g2 = (l0 - l2) / eps
                        grad_logl[it.multi_index] = g1
                        hess_logl[it.multi_index] = (g1 - g2) / eps
                        xv += eps

                gradl0 = grad_logl
        else:
            logl0 = self.logpdf(x)
            logl1 = self.logpdf(x + eps)
            logl2 = self.logpdf(x - eps)
            gradl0 = (logl1 - logl0) / eps
            gradl1 = (logl0 - logl2) / eps
            hess_logl = (gradl0 - gradl1) / eps

        return logl0, gradl0, hess_logl

    logpdf_gradient = numerical_logpdf_gradient
    logpdf_gradient_hessian = numerical_logpdf_gradient_hessian

    @classmethod
    def project(
            cls, samples: np.ndarray, log_weight_list: Optional[np.ndarray] = None, **kwargs
    ) -> "AbstractMessage":
        """Calculates the sufficient statistics of a set of samples
        and returns the distribution with the appropriate parameters
        that match the sufficient statistics
        """
        # if weight_list aren't passed then equally weight all samples

        # Numerically stable weighting for very small/large weight_list

        # rescale coordinates to 'natural parameter space'
        if log_weight_list is None:
            log_weight_list = np.zeros_like(samples)

        log_w_max = np.max(log_weight_list, axis=0, keepdims=True)
        w = np.exp(log_weight_list - log_w_max)
        norm = w.mean(0)
        log_norm = np.log(norm) + log_w_max[0]

        tx = cls.to_canonical_form(samples)
        w /= norm
        suff_stats = (tx * w[None, ...]).mean(1)

        assert np.isfinite(suff_stats).all()

        cls_ = cls._projection_class or cls._Base_class or cls
        return cls_.from_sufficient_statistics(suff_stats, log_norm=log_norm, **kwargs)

    @classmethod
    def from_mode(
            cls, mode: np.ndarray, covariance: np.ndarray, **kwargs
    ) -> "AbstractMessage":
        pass

    def log_normalisation(self, *elems: Union["AbstractMessage", float]) -> np.ndarray:
        """
        Calculates the log of the integral of the product of a
        set of distributions

        NOTE: ignores log normalisation
        """
        # Remove floats from messages passed
        dists: List[AbstractMessage] = [
            dist
            for dist in self._iter_dists(elems)
            if isinstance(dist, AbstractMessage)
        ]

        # Calculate log product of message normalisation
        log_norm = self.log_base_measure - self.log_partition
        log_norm += sum(dist.log_base_measure - dist.log_partition for dist in dists)

        # Calculate log normalisation of product of messages
        prod_dist = self.sum_natural_parameters(*dists)
        log_norm -= prod_dist.log_base_measure - prod_dist.log_partition

        return log_norm

    @staticmethod
    def _iter_dists(dists) -> Iterator[Union["AbstractMessage", float]]:
        for elem in dists:
            if isinstance(elem, Prior):
                yield elem
            elif np.isscalar(elem):
                yield elem
            else:
                for dist in elem:
                    yield dist

    def update_invalid(self, other: "AbstractMessage") -> "AbstractMessage":
        valid = self.check_valid()
        if self.ndim:
            valid_parameters: Iterator[np.ndarray] = (
                np.where(valid, p, p_safe) for p, p_safe in zip(self, other)
            )
        else:
            # TODO: Fairly certain this would not work
            valid_parameters = iter(self if valid else other)
        cls = self._Base_class or type(self)
        new = cls(
            *valid_parameters,
            log_norm=self.log_norm,
            id_=self.id,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
        )
        return new

    def check_support(self) -> np.ndarray:
        if self._parameter_support is not None:
            return reduce(
                and_,
                (
                    (p >= support[0]) & (p <= support[1])
                    for p, support in zip(self.parameters, self._parameter_support)
                ),
            )
        elif self.ndim:
            return np.array(True, dtype=bool, ndmin=self.ndim)
        return np.array([True])

    def check_finite(self) -> np.ndarray:
        return np.isfinite(self.natural_parameters).all(0)

    def check_valid(self) -> np.ndarray:
        return self.check_finite() & self.check_support()

    @cached_property
    def is_valid(self) -> Union[np.ndarray, np.bool_]:
        return np.all(self.check_finite()) and np.all(self.check_support())

    @staticmethod
    def _get_mean_variance(
            mean: np.ndarray, covariance: np.ndarray
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
                f"must be (), {mean.shape}, or {mean.shape * 2}"
            )
        return mean, variance

    def __call__(self, x):
        return np.sum(self.logpdf(x))

    def factor_jacobian(
            self, x: np.ndarray, _variables: Optional[Tuple[str]] = ("x",)
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, ...]]]:
        loglike, g = self.logpdf_gradient(x)
        g = np.expand_dims(g, list(range(loglike.ndim)))
        return loglike.sum(), (g,)

    def as_factor(self, variable: "Variable", name: Optional[str] = None):
        from autofit.graphical.factor_graphs import Factor

        if name is None:
            shape = self.shape
            clsname = type(self).__name__
            family = clsname[:-7] if clsname.endswith("Message") else clsname
            name = f"{family}Likelihood" + (str(shape) if shape else "")

        return Factor(
            self,
            variable,
            name=name,
            factor_jacobian=self.factor_jacobian,
            plates=variable.plates,
            arg_names=["x"],
        )

    def calc_exact_update(self, x: "AbstractMessage") -> "AbstractMessage":
        return self,

    def has_exact_projection(self, x: "AbstractMessage") -> bool:
        return type(self) is type(x)

    @classmethod
    def transformed(
            cls,
            transform: Union[AbstractDensityTransform, Type[AbstractDensityTransform]],
            clsname: Optional[str] = None,
            support: Optional[Tuple[Tuple[float, float], ...]] = None,
            wrapper_cls=None,
    ):
        # noinspection PyUnresolvedReferences
        """
        transforms the distribution according the passed transform,
        returns a newly created class that encodes the transformation.

        Parameters
        ----------
        wrapper_cls
        transform: AbstractDensityTransform
            object that transforms the density
        clsname: str, optional
            the class name of the newly created class.
            defaults to "Transformed<OriginalClassName>"
        support: Tuple[Tuple[float, float], optional
            the support of the new class. Generally this can be
            automatically calculated from the parent class

        Examples
        --------
        >>> from autofit.messages.normal import NormalMessage

        Normal distributions have infinite univariate support
        >>> NormalMessage._support
        ((-inf, inf),)

        We can tranform the NormalMessage to the unit interval
        using `transform.phi_transform`
        >>> UnitNormal = NormalMessage.transformed(transform.phi_transform)
        >>> message = UnitNormal(1.2, 0.8)
        >>> message._support
        ((0.0, 1.0),)

        Samples from the UnitNormal will exist in the Unit interval
        >>> samples = message.sample(1000)
        >>> samples.min(), samples.mean(), samples.max()
        (0.06631750944045942, 0.8183189295040845, 0.9999056316923468)

        Projections still work for the transformed class
        >>> UnitNormal.project(samples, samples*0)
        TransformedNormalMessage(mu=1.20273342, sigma=0.80929032)

        Can specify the name of the new transformed class
        >>> NormalMessage.transformed(transform.phi_transform, 'UnitNormal')(0, 1.)
        UnitNormal(mu=0, sigma=1.)

        The transformed objects are pickleable
        >>> import pickle
        >>> pickle.loads(pickle.dumps(message))
        TransformedNormalMessage(mu=1.2, sigma=0.8)

        The transformed objects also are normalised,
        >>> from scipy.integrate import quad
        >>> # noinspection PyTypeChecker
        >>> quad(message.pdf, 0, 1)
        (1.0000000000114622, 3.977073226302252e-09)

        Can also nest transforms
        >>> WeirdNormal = NormalMessage.transformed(
            transform.log_transform).transformed(
            transform.exp_transform)
        This transformation is equivalent to the identity transform!
        >>> WeirdNormal.project(NormalMessage(0.3, 0.8).sample(1000))
        Transformed2NormalMessage(mu=0.31663248, sigma=0.79426984)

        This functionality is more useful for applying linear shifts
        e.g.
        >>> ShiftedUnitNormal = NormalMessage.transformed(
            transform.phi_transform
        ).shifted(shift=0.7, scale=2.3)
        >>> ShiftedUnitNormal._support
        ((0.7, 3.0),)
        >>> samples = ShiftedUnitNormal(0.2, 0.8).sample(1000)
        >>> samples.min(), samples.mean(), samples.max()
        """
        from .transform_wrapper import TransformedWrapper

        wrapper_cls = wrapper_cls or TransformedWrapper
        return wrapper_cls(
            cls=cls,
            transform=transform,
            clsname=clsname,
            support=support,
        )

    @classmethod
    def shifted(
            cls,
            shift: float = 0,
            scale: float = 1,
            wrapper_cls=None,
    ):
        return cls.transformed(
            LinearShiftTransform(shift=shift, scale=scale),
            clsname=f"Shifted{cls.__name__}",
            wrapper_cls=wrapper_cls,
        )

    @classmethod
    def _reconstruct(
            cls,
            parameters: Tuple[np.ndarray, ...],
            log_norm: float,
            id_,
            lower_limit,
            upper_limit,
            *args,
    ):
        return cls(
            *parameters,
            log_norm=log_norm,
            id_=id_,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

    def __reduce__(self):
        # serialises TransformedMessage during pickling
        return (
            self._reconstruct,
            (
                self.parameters,
                self.log_norm,
                self.id,
                self.lower_limit,
                self.upper_limit,
            ),
        )

    def _sample(self, n_samples):
        # Needed for nested TransformedMessage method resolution
        return self.sample(n_samples)

    @classmethod
    def _logpdf_gradient(cls, self, x):
        # Needed for nested TransformedMessage method resolution
        return cls.logpdf_gradient(self, x)

    @classmethod
    def _logpdf_gradient_hessian(cls, self, x):
        # Needed for nested TransformedMessage method resolution
        return cls.logpdf_gradient_hessian(self, x)


def map_dists(
        dists: Dict[str, AbstractMessage],
        values: Dict[str, np.ndarray],
        _call: str = "logpdf",
) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Calls a method (default: logpdf) for each Message in dists
    on the corresponding value in values
    """
    for v in dists.keys() & values.keys():
        dist = dists[v]
        if isinstance(dist, AbstractMessage):
            yield v, getattr(dist, _call)(values[v])
