import functools
from typing import Tuple, Optional

import numpy as np

from autoconf import cached_property
from autofit.messages.abstract import MessageInterface
from autofit.messages.transform import AbstractDensityTransform


def arithmetic(func):
    @functools.wraps(func)
    def wrapper(self, other):
        if isinstance(other, TransformedMessage):
            other = other.base_message
        return self.with_base(func(self, other))

    return wrapper


class TransformedMessage(MessageInterface):
    @property
    def shape(self):
        return self.base_message.shape

    def __init__(
        self, base_message, *transforms: AbstractDensityTransform, id_=None,
    ):
        while isinstance(base_message, TransformedMessage):
            transforms += base_message.transforms
            base_message = base_message.base_message

        self.transforms = transforms
        self.base_message = base_message
        self.id = id_

    def __call__(self, *args, **kwargs):
        kwargs["id_"] = kwargs.get("id_") or self.id
        return self.with_base(type(self.base_message)(*args, **kwargs))

    def copy(self):
        return TransformedMessage(self.base_message, *self.transforms, id_=self.id)

    def with_base(self, message):
        return TransformedMessage(message, *self.transforms, id_=self.id)

    @arithmetic
    def __mul__(self, other):
        return self.base_message * other

    @arithmetic
    def __pow__(self, other):
        return self.base_message ** other

    def __rmul__(self, other):
        return self.base_message * other

    @arithmetic
    def __truediv__(self, other):
        return self.base_message / other

    @arithmetic
    def __sub__(self, other):
        return self.base_message - other

    def project(
        self, samples, log_weight_list, **_,
    ):
        return TransformedMessage(
            self.base_message.project(samples, log_weight_list),
            *self.transforms,
            id_=self.id,
        )

    @property
    def natural_parameters(self):
        return self.base_message.natural_parameters

    def sample(self, n_samples: Optional[int] = None):
        x = self.base_message.sample(n_samples)
        return self.inverse_transform(x)

    def transform(self, x):
        for transform in self.transforms:
            x = transform.transform(x)
        return x

    def inverse_transform(self, x):
        for transform in reversed(self.transforms):
            x = transform.inv_transform(x)
        return x

    def transform_det(self, x):
        for transform in self.transforms:
            x = transform.log_det(x)
        return x

    def invert_natural_parameters(
        self, natural_parameters: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        return self.base_message.invert_natural_parameters(natural_parameters)

    def cdf(self, x):
        return self.base_message.cdf(self.transform(x))

    @property
    def log_partition(self) -> np.ndarray:
        return self.base_message.log_partition

    def invert_sufficient_statistics(self, sufficient_statistics):
        return self.base_message.invert_sufficient_statistics(sufficient_statistics)

    def value_for(self, unit):
        return self.inverse_transform(self.base_message.value_for(unit))

    def calc_log_base_measure(self, x) -> np.ndarray:
        x = self.transform(x)
        return self.base_message.calc_log_base_measure(x)

    def to_canonical_form(self, x) -> np.ndarray:
        x = self.transform(x)
        return self.base_message.to_canonical_form(x)

    @cached_property
    def mean(self) -> np.ndarray:
        return self.inverse_transform(self.base_message.mean)

    # @cached_property
    # @property
    # def variance(self) -> np.ndarray:
    #     # noinspection PyUnresolvedReferences
    #     jac = self._transform.jacobian(self.mean)
    #     return jac.quad(self._Message.variance.func(self))

    def _sample(self, n_samples) -> np.ndarray:
        x = self.base_message._sample(n_samples)
        return self.inverse_transform(x)

    def _factor(self, _, x: np.ndarray,) -> np.ndarray:
        x, log_det = self.transform_det(x)
        eta = self.base_message._broadcast_natural_parameters(x)
        t = self.base_message.to_canonical_form(x)
        log_base = self.calc_log_base_measure(x) + log_det
        return self.base_message.natural_logpdf(eta, t, log_base, self.log_partition)

    # def _factor_gradient(self, _, x: np.ndarray,) -> np.ndarray:
    #     x, logd, logd_grad, jac = cls._transform.transform_det_jac(x)
    #     logl, grad = cls._Message._logpdf_gradient(self, x)
    #     return logl + logd, grad * jac + logd_grad

    def factor(self, x):
        return self._factor(self, x)

    # def factor_gradient(self, x):
    #     return self._factor_gradient(self, x)

    @classmethod
    def _logpdf_gradient(  # type: ignore
        cls, self, x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, jac = cls._transform.transform_jac(x)
        logl, grad = cls._Message._logpdf_gradient(self, x)
        return logl, grad * jac

    def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        jacobians = []
        for transform in self.transforms:
            x, jacobian = transform.transform_jac(x)
            jacobians.append(jacobian)

        log_likelihood, gradient = self.base_message.logpdf_gradient(x)

        for jacobian in reversed(jacobians):
            gradient = gradient * jacobian

        return log_likelihood, gradient
