import inspect
from typing import Type, Tuple, Union

import numpy as np

from .abstract import AbstractMessage
from .transform import AbstractDensityTransform
from ..tools.cached_property import cached_property


class TransformedMessage(AbstractMessage):
    _Message: Type[AbstractMessage]
    _transform: Union[AbstractDensityTransform, Type[AbstractDensityTransform]]
    _depth = 0

    def __init__(self, *args, **kwargs):
        if inspect.isclass(self._transform):
            transform_args = inspect.getfullargspec(
                self._transform
            ).args[1:]

            transform_dict = dict()
            for arg in transform_args:
                if arg in kwargs:
                    transform_dict[
                        arg
                    ] = kwargs.pop(arg)
            self._transform = self._transform(
                **transform_dict
            )

        self.instance = self._Message(*args, **kwargs)
        super().__init__(
            *self.instance.parameters,
            log_norm=self.instance.log_norm,
            lower_limit=self.instance.lower_limit,
            upper_limit=self.instance.upper_limit,
            id_=self.instance.id
        )

    @property
    def natural_parameters(self):
        return self.instance.natural_parameters

    @property
    def log_partition(self) -> np.ndarray:
        return self.instance.log_partition

    def __getattr__(self, item):
        if item == "__setstate__":
            raise AttributeError()
        return getattr(
            self.instance,
            item
        )

    @classmethod
    def invert_natural_parameters(
            cls,
            natural_parameters
    ):
        return cls._Message.invert_natural_parameters(
            natural_parameters
        )

    def invert_sufficient_statistics(
            self,
            sufficient_statistics
    ):
        return self.instance.invert_sufficient_statistics(
            sufficient_statistics
        )

    def value_for(self, unit):
        return self._transform.inv_transform(
            self.instance.value_for(
                unit
            )
        )

    # noinspection PyMethodOverriding
    @classmethod
    def _reconstruct(  # type: ignore
            cls,
            Message: 'AbstractMessage',
            clsname: str,
            transform: AbstractDensityTransform,
            parameters: Tuple[np.ndarray, ...],
            log_norm: float,
            id_,
            lower_limit,
            upper_limit
    ):
        # Reconstructs TransformedMessage during unpickling
        Transformed = Message.transformed(transform, clsname)
        return Transformed(
            *parameters,
            log_norm=log_norm,
            id_=id_,
            lower_limit=lower_limit,
            upper_limit=upper_limit
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
                self.id,
                self.lower_limit,
                self.upper_limit
            ),
        )

    def calc_log_base_measure(self, x) -> np.ndarray:
        x, log_det = self._transform.transform_det(x)
        log_base = self.instance.calc_log_base_measure(x)
        return log_base + log_det

    def to_canonical_form(self, x) -> np.ndarray:
        x = self._transform.transform(x)
        return self.instance.to_canonical_form(x)

    @cached_property
    def mean(self) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self._transform.inv_transform(self._Message.mean.func(self))

    @cached_property
    def variance(self) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self._transform.inv_transform(self._Message.variance.func(self))

    def _sample(self, n_samples) -> np.ndarray:
        x = self.instance._sample(n_samples)
        return self._transform.inv_transform(x)

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
        return self._sample(n_samples)

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
