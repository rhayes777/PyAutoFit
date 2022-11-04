from typing import Tuple, Optional

import numpy as np

from autoconf import cached_property
from autofit.messages.transform import AbstractDensityTransform


class TransformedMessage:
    def __new__(cls, base_message, *transforms: AbstractDensityTransform):
        if isinstance(base_message, TransformedMessage):
            return TransformedMessage(
                base_message.base_message, *(base_message.transforms + transforms)
            )
        return object.__new__(TransformedMessage)

    def __init__(
        self, base_message, *transforms: AbstractDensityTransform,
    ):
        self.transforms = transforms
        self.base_message = base_message

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

    # @classmethod
    # def _logpdf_gradient(  # type: ignore
    #     cls, self, x: np.ndarray,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     x, jac = cls._transform.transform_jac(x)
    #     logl, grad = cls._Message._logpdf_gradient(self, x)
    #     return logl, grad * jac

    # def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     return self._logpdf_gradient(self, x)
