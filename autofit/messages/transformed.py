from typing import Type, Tuple

import numpy as np

from .abstract import AbstractMessage
from .transform import AbstractDensityTransform
from ..tools.cached_property import cached_property


class TransformedMessage(AbstractMessage):
    _Message: Type[AbstractMessage]
    _transform: AbstractDensityTransform
    _depth = 0

    # noinspection PyMethodOverriding
    @classmethod
    def _reconstruct(  # type: ignore
            cls,
            Message: 'AbstractMessage',
            clsname: str,
            transform: AbstractDensityTransform,
            parameters: Tuple[np.ndarray, ...],
            log_norm: float,
            id_
    ):
        # Reconstructs TransformedMessage during unpickling
        Transformed = Message.transformed(transform, clsname)
        return Transformed(
            *parameters,
            log_norm=log_norm,
            id_=id_
        )

    def __reduce__(self):
        # serialises TransformedMessage during pickling
        return (
            TransformedMessage._reconstruct,
            (
                self._Message,
                self.__class__.__name__,
                self._transform,
                self.parameters,
                self.log_norm,
                self.id
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
        # noinspection PyUnresolvedReferences
        return self._transform.inv_transform(self._Message.mean.func(self))

    @cached_property
    def variance(self) -> np.ndarray:
        # noinspection PyUnresolvedReferences
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
