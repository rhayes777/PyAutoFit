from abc import ABC, abstractmethod
from functools import reduce, wraps
from inspect import getfullargspec
from itertools import count
from numbers import Real
from operator import and_
from typing import Optional, Tuple, Union, Iterator, Type, List

import numpy as np

from autofit.mapper.prior.prior import Prior, GaussianPrior
from .transform import AbstractDensityTransform, LinearShiftTransform
from ..factor_graphs.jacobians import FactorJacobian
from ..utils import cached_property
from ...mapper.variable import Variable


def assert_ids_match(func):
    @wraps(func)
    def wrapper(self, other):
        if isinstance(
                other,
                AbstractMessage
        ) and self.id != other.id:
            raise AssertionError(
                f"Message with id {self.id} should not be compared to message with id {other.id}"
            )
        result = func(self, other)
        return result

    return wrapper


class AbstractMessage(ABC):
    log_base_measure: float
    _projection_class: Optional[Type['AbstractMessage']] = None
    _multivariate: bool = False
    _parameter_support: Optional[Tuple[Tuple[float, float], ...]] = None
    _support: Optional[Tuple[Tuple[float, float], ...]] = None

    _ids = count()

    def __init__(
            self,
            *parameters: Union[np.ndarray, float],
            log_norm=0.,
            id_=None
    ):
        self.id_ = id_ or next(self._ids)
        self.log_norm = log_norm
        self._broadcast = np.broadcast(*parameters)
        self.id = id_
        if self.shape:
            self.parameters = tuple(
                np.asanyarray(p) for p in parameters)
        else:
            self.parameters = tuple(parameters)

    @classmethod
    def from_prior(cls, prior: Prior) -> "AbstractMessage":
        from autofit.graphical import NormalMessage
        if isinstance(
                prior, GaussianPrior
        ):
            return NormalMessage(
                mu=prior.mean,
                sigma=prior.sigma,
                id_=prior.id
            )
        raise TypeError(
            f"No message exists for prior of type {type(prior)}"
        )

    @cached_property
    @abstractmethod
    def natural_parameters(self):
        pass

    @abstractmethod
    def sample(self, n_samples: Optional[int] = None):
        pass

    @staticmethod
    @abstractmethod
    def invert_natural_parameters(natural_parameters: np.ndarray
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
    def mean(self) -> np.ndarray:
        pass

    @cached_property
    @abstractmethod
    def variance(self) -> np.ndarray:
        pass

    @cached_property
    def scale(self) -> np.ndarray:
        return self.variance ** 0.5

    def __hash__(self):
        return self.id or super().__hash__()

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
    def from_natural_parameters(
            cls,
            natural_parameters: np.ndarray,
            **kwargs
    ) -> "AbstractMessage":
        cls = cls._projection_class or cls
        args = cls.invert_natural_parameters(natural_parameters)
        return cls(*args, **kwargs)

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
        cls = cls._projection_class or cls
        return cls.from_natural_parameters(natural_params, **kwargs)

    def sum_natural_parameters(
            self, *dists: "AbstractMessage"
    ) -> "AbstractMessage":
        """return the unnormalised result of multiplying the pdf
        of this distribution with another distribution of the same
        type
        """
        new_params = sum(
            (dist.natural_parameters for dist in self._iter_dists(dists)
             if isinstance(dist, AbstractMessage)),
            self.natural_parameters)
        mul_dist = self.from_natural_parameters(
            new_params,
            id_=self.id
        )
        return mul_dist

    @assert_ids_match
    def sub_natural_parameters(
            self, other: "AbstractMessage"
    ) -> "AbstractMessage":
        """return the unnormalised result of dividing the pdf
        of this distribution with another distribution of the same
        type"""
        log_norm = self.log_norm - other.log_norm
        new_params = (self.natural_parameters - other.natural_parameters)
        div_dist = self.from_natural_parameters(
            new_params,
            log_norm=log_norm,
            id_=self.id
        )
        return div_dist

    _multiply = sum_natural_parameters
    _divide = sub_natural_parameters

    @assert_ids_match
    def __mul__(self, other: Union["AbstractMessage", Real]) -> "AbstractMessage":
        if isinstance(other, AbstractMessage):
            return self._multiply(other)
        else:
            log_norm = self.log_norm + np.log(other)
            return type(self)(
                *self.parameters,
                log_norm=log_norm,
                id_=self.id
            )

    @assert_ids_match
    def __rmul__(self, other: "AbstractMessage") -> "AbstractMessage":
        return self * other

    @assert_ids_match
    def __truediv__(self, other: Union["AbstractMessage", Real]) -> "AbstractMessage":
        if isinstance(other, AbstractMessage):
            return self._divide(other)
        else:
            log_norm = self.log_norm - np.log(other)
            return type(self)(
                *self.parameters,
                log_norm=log_norm,
                id_=self.id
            )

    @assert_ids_match
    def __pow__(self, other: Real) -> "AbstractMessage":
        natural = self.natural_parameters
        new_params = other * natural
        log_norm = other * self.log_norm
        return self.from_natural_parameters(
            new_params,
            log_norm=log_norm,
            id_=self.id
        )

    @classmethod
    def parameter_names(cls):
        return getfullargspec(cls.__init__).args[1:-1]

    def __str__(self) -> str:
        param_attrs = [
            (attr, np.asanyarray(getattr(self, attr)))
            for attr in self.parameter_names()]
        if self.shape:
            pad = max(len(attr) for attr, _ in param_attrs)
            attr_str = "    {:<%d}={}" % pad
            param_strs = ',\n'.join(
                attr_str.format(
                    attr, np.array2string(val, prefix=' ' * (pad + 5)))
                for attr, val in param_attrs)
            return f"{type(self).__name__}(\n{param_strs})"
        else:
            param_strs = ', '.join(
                attr + '=' +
                np.array2string(val, prefix=' ' * (len(attr) + 1))
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
            log_base = self.calc_log_base_measure(x)
            # TODO this can be made more efficient using tensordot
            eta_t = np.multiply(eta, t).sum(0)
            return log_base + eta_t - self.log_partition

        elif shape[1:] == self.shape:
            eta = self.natural_parameters
            t = self.to_canonical_form(x)
            log_base = self.calc_log_base_measure(x)
            eta_t = np.multiply(eta[:, None, ...], t).sum(0)
            return log_base + eta_t - self.log_partition

        raise ValueError(
            f"shape of passed value {shape} does not "
            f"match message shape {self.shape}")

    def numerical_logpdf_gradient(self, x: np.ndarray, eps: float = 1e-6
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
                        grad_logl[sl + it.multi_index] = (logl - logl0) / eps
                        xv -= eps
            else:
                l0 = logl0.sum()
                grad_logl = np.empty_like(x0)
                with np.nditer(x0, flags=['multi_index'], op_flags=['readwrite']) as it:
                    for xv in it:
                        xv += eps
                        logl = self.logpdf(x0).sum()  # type: ignore
                        grad_logl[it.multi_index] = (logl - l0) / eps
                        xv -= eps
        else:
            logl0 = self.logpdf(x)
            grad_logl = (self.logpdf(x + eps) - logl0) / eps

        return logl0, grad_logl

    def numerical_logpdf_gradient_hessian(self, x: np.ndarray, eps: float = 1e-6
                                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                        hess_logl[sl + it.multi_index] = (gradl - gradl0) / eps
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
            cls,
            samples: np.ndarray,
            log_weight_list: Optional[np.ndarray] = None,
            id_=None
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

        cls = cls._projection_class or cls
        return cls.from_sufficient_statistics(
            suff_stats,
            log_norm=log_norm,
            id_=id_
        )

    def has_exact_projection(self, x: "AbstractMessage") -> bool:
        return type(self) == type(x)

    def calc_exact_projection(self, x: "AbstractMessage"):
        if type(self) == type(x):
            projection = self * x
            projection.log_norm = self.log_normalisation(x)
            return {'x': projection}
        raise NotImplementedError()

    def calc_exact_update(self, x: "AbstractMessage"):
        if type(self) == type(x):
            log_norm = self.log_normalisation(x)
            return {'x': type(x)(*self.parameters, log_norm=log_norm)}
        else:
            raise NotImplementedError()

    @classmethod
    def from_mode(
            cls,
            mode: np.ndarray,
            covariance: np.ndarray,
            id_
    ) -> "AbstractMessage":
        pass

    def log_normalisation(self, *elems: Union["AbstractMessage", float]
                          ) -> np.ndarray:
        """
        Calculates the log of the integral of the product of a
        set of distributions

        NOTE: ignores log normalisation
        """
        # Remove floats from messages passed
        dists: List[AbstractMessage] = [
            dist for dist in self._iter_dists(elems)
            if isinstance(dist, AbstractMessage)]

        # Calculate log product of message normalisation
        log_norm = self.log_base_measure - self.log_partition
        log_norm += sum(
            dist.log_base_measure - dist.log_partition for dist in dists)

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
            valid_parameters: Iterator[np.ndarray] = (
                np.where(valid, p, p_safe) for p, p_safe in zip(self, other))
        else:
            # TODO: Fairly certain this would not work
            valid_parameters = iter(self if valid else other)
        return type(self)(
            *valid_parameters,
            log_norm=self.log_norm,
            id_=self.id
        )

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

    @cached_property
    def is_valid(self) -> np.bool_:
        return np.all(self.check_finite()) and np.all(self.check_support())

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

    def __call__(
            self,
            x: np.ndarray,
            _variables: Optional[Tuple[str]] = ('x',)
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, ...]]]:
        if _variables is None:
            return self.logpdf(x)
        else:
            if 'x' in _variables:
                loglike, g = self.logpdf_gradient(x)
                g = np.expand_dims(g, list(range(loglike.ndim)))
                return loglike, (g,)
            else:
                return self.logpdf(x), ()

    def as_factor(self, variable: "Variable", name: Optional[str] = None
                  ) -> "FactorJacobian":
        from autofit.graphical import FactorJacobian
        if name is None:
            shape = self.shape
            clsname = type(self).__name__
            family = clsname[:-7] if clsname.endswith('Message') else clsname
            name = f"{family}Likelihood" + (str(shape) if shape else '')

        return FactorJacobian(self, x=variable, name=name, vectorised=True)

    @classmethod
    def transformed(
            cls,
            transform: AbstractDensityTransform,
            clsname: Optional[str] = None,
            support: Optional[Tuple[Tuple[float, float], ...]] = None,
    ) -> Type["AbstractMessage"]:
        """
        transforms the distribution according the passed transform, 
        returns a newly created class that encodes the transformation.

        Parameters
        ----------
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
        >>> from autofit.graphical.messages import NormalMessage, transform

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
        support = support or tuple(zip(*map(
            transform.inv_transform, map(np.array, zip(*cls._support))
        ))) if cls._support else cls._support
        projectionClass = (
            None if cls._projection_class is None
            else cls._projection_class.transformed(transform)
        )
        if issubclass(cls, TransformedMessage):
            depth = cls._depth + 1
            clsname = clsname or f"Transformed{depth}{cls._Message.__name__}"

            # Don't doubly inherit if transforming already transformed message
            class Transformed(cls):  # type: ignore
                __qualname__ = clsname
                _Message = cls
                _transform = transform
                _support = support
                __projection_class = projectionClass
                _depth = depth

        else:
            clsname = clsname or f"Transformed{cls.__name__}"

            class Transformed(TransformedMessage, cls):  # type: ignore
                __qualname__ = clsname
                _Message = cls
                _transform = transform
                _support = support
                __projection_class = projectionClass
                parameter_names = cls.parameter_names
                _depth = 1

        Transformed.__name__ = clsname
        return Transformed

    @classmethod
    def shifted(cls, shift: float = 0, scale: float = 1) -> Type["AbstractMessage"]:
        return cls.transformed(
            LinearShiftTransform(shift=shift, scale=scale),
            clsname=f"Shifted{cls.__name__}"
        )

    @classmethod
    def _reconstruct(
            cls,
            parameters: Tuple[np.ndarray, ...],
            log_norm: float,
            *args
    ):
        return cls(*parameters, log_norm=log_norm)

    def __reduce__(self):
        # serialises TransformedMessage during pickling
        return (
            self._reconstruct,
            (
                self.parameters,
                self.log_norm
            ),
        )

    @classmethod
    def _sample(cls, self, n_samples):
        # Needed for nested TransformedMessage method resolution
        return cls.sample(self, n_samples)

    @classmethod
    def _logpdf_gradient(cls, self, x):
        # Needed for nested TransformedMessage method resolution
        return cls.logpdf_gradient(self, x)

    def as_prior(self):
        raise NotImplemented()


class TransformedMessage(AbstractMessage):
    _Message: Type[AbstractMessage]
    _transform: AbstractDensityTransform
    _depth = 0

    @classmethod
    def _reconstruct(  # type: ignore
            cls,
            Message: 'AbstractMessage',
            clsname: str,
            transform: AbstractDensityTransform,
            parameters: Tuple[np.ndarray, ...],
            log_norm: float
    ):
        # Reconstructs TransformedMessage during unpickling
        Transformed = Message.transformed(transform, clsname)
        return Transformed(*parameters, log_norm=log_norm)

    def __reduce__(self):
        # serialises TransformedMessage during pickling
        return (
            TransformedMessage._reconstruct,
            (
                self._Message,
                self.__class__.__name__,
                self._transform,
                self.parameters,
                self.log_norm
            ),
        )

    @classmethod
    def calc_log_base_measure(cls, x) -> np.ndarray:
        x, log_det = cls._transform.transform_det(x)
        log_base = cls._Message.calc_log_base_measure(x)
        return log_base + log_det

    @classmethod
    def to_canonical_form(cls, x) -> np.ndarray:
        x = cls._transform.transform(x)
        return cls._Message.to_canonical_form(x)

    @cached_property
    def mean(self) -> np.ndarray:
        return self._transform.inv_transform(self._Message.mean.func(self))

    @cached_property
    def variance(self) -> np.ndarray:
        return self._transform.inv_transform(self._Message.variance.func(self))

    @classmethod
    def _sample(cls, self, n_samples) -> np.ndarray:
        x = cls._Message._sample(self, n_samples)
        return cls._transform.inv_transform(x)

    @classmethod
    def _logpdf_gradient(  # type: ignore
            cls,
            self,
            x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, logd, logd_grad, jac = cls._transform.transform_det_jac(x)
        logl, grad = cls._Message._logpdf_gradient(self, x)
        return logl + logd, grad * jac + logd_grad

    def sample(self, n_samples=None) -> np.ndarray:
        return self._sample(self, n_samples)

    def logpdf_gradient(
            self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._logpdf_gradient(self, x)

    # TODO add code for analytic hessians when Jacobian is fixed e.g. for shifted messages
    logpdf_gradient_hessian = AbstractMessage.numerical_logpdf_gradient_hessian

    @classmethod
    def from_mode(
            cls,
            mode: np.ndarray,
            covariance: np.ndarray,
            id_=None
    ) -> "AbstractMessage":
        mode, jac = cls._transform.transform_jac(mode)
        covariance = jac.invquad(covariance)
        return cls.from_mode(mode, covariance, id_=id_)
