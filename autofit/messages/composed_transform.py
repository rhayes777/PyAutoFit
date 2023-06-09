import functools
from typing import Tuple, Optional, Union

import numpy as np

from autofit.messages.abstract import MessageInterface
from autofit.messages.transform import AbstractDensityTransform


def arithmetic(func):
    """
    When arithmetic is performed between a two transformed messages the
    operation is performed between the base messages and the result it
    encapsulated in a transformed message with the same set of transforms.
    """

    @functools.wraps(func)
    def wrapper(self, other):
        if isinstance(other, TransformedMessage):
            other = other.base_message
        return self.with_base(func(self, other))

    return wrapper


def transform(func):
    """
    Decorator to transform the function argument in the space of the
    transformed message to the space of the underlying message.

    For example, a UniformPrior with limits 10 and 20 could be passed
    a value 15. If the underlying message is a NormalMessage with a
    mean of 0 then the result would be 0.
    """

    @functools.wraps(func)
    def wrapper(self, x):
        x = self._transform(x)
        return func(self, x)

    return wrapper


def inverse_transform(func):
    """
    Decorator to transform the result of a function in the space of
    the base message to a value in the space of the transformed message.

    Inverts transform (above)
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        return self._inverse_transform(result)

    return wrapper


class TransformedMessage(MessageInterface):
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        base_message: MessageInterface,
        *transforms: AbstractDensityTransform,
        id_: Optional[int] = None,
        lower_limit=float("-inf"),
        upper_limit=float("inf"),
    ):
        """
        Comprises a base message such as a normal message and a list of transforms
        that transform the message into some new distribution, for example the
        shifted uniform distribution which underpins a UniformPrior.

        Parameters
        ----------
        base_message
            A message
        transforms
            A list of transforms applied left to right. For example, a shifted uniform
            normal message is first converted to uniform normal then shifted
        id_
        lower_limit
        upper_limit
        """
        while isinstance(base_message, TransformedMessage):
            transforms = base_message.transforms + transforms
            base_message = base_message.base_message

        self.transforms = transforms
        self.base_message = base_message
        self.id = id_

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def from_natural_parameters(self, new_params, **kwargs):
        return self.with_base(
            self.base_message.from_natural_parameters(new_params, **kwargs,)
        )

    @property
    def broadcast(self):
        return self.base_message.broadcast

    def check_support(self) -> np.ndarray:
        return self.base_message.check_support()

    def __call__(self, *args, **kwargs):
        kwargs["id_"] = kwargs.get("id_") or self.id
        return self.with_base(type(self.base_message)(*args, **kwargs))

    def copy(self):
        return TransformedMessage(self.base_message, *self.transforms, id_=self.id)

    def with_base(self, message: MessageInterface) -> "TransformedMessage":
        """
        Creates a new TransformedMessage with the same id and transforms but a new
        underlying base message
        """
        return TransformedMessage(message, *self.transforms, id_=self.id)

    @arithmetic
    def __mul__(self, other):
        return self.base_message * other

    @arithmetic
    def __pow__(self, other):
        return self.base_message ** other

    @arithmetic
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

    def kl(self, dist):
        return self.base_message.kl(dist.base_message)

    @property
    def natural_parameters(self):
        return self.base_message.natural_parameters

    @inverse_transform
    def sample(self, n_samples: Optional[int] = None):
        return self.base_message.sample(n_samples)

    def _transform(self, x: float) -> float:
        """
        Transform some value in the space of the transformed message to
        the space of the underlying message.

        For example, a UniformPrior with limits 10 and 20 could be passed
        a value 15. If the underlying message is a NormalMessage with a
        mean of 0 then the result would be 0.

        Parameters
        ----------
        x
            A value in the space of the transformed message

        Returns
        -------
        A value in the space of the base message
        """
        for _transform in reversed(self.transforms):
            x = _transform.transform(x)
        return x

    def _inverse_transform(self, x: float) -> float:
        """
        Transform some value in the space of the base message to a value in
        the space of the transformed message.

        Inverts transform (above)
        """
        for _transform in self.transforms:
            x = _transform.inv_transform(x)
        return x

    def transform_det(self, x):
        for _transform in self.transforms:
            x = _transform.log_det(x)
        return x

    def invert_natural_parameters(
        self, natural_parameters: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        return self.base_message.invert_natural_parameters(natural_parameters)

    @transform
    def cdf(self, x):
        return self.base_message.cdf(x)

    @property
    def log_partition(self) -> np.ndarray:
        return self.base_message.log_partition

    def invert_sufficient_statistics(self, sufficient_statistics):
        return self.base_message.invert_sufficient_statistics(sufficient_statistics)

    @inverse_transform
    def value_for(self, unit):
        return self.base_message.value_for(unit)

    @transform
    def calc_log_base_measure(self, x) -> np.ndarray:
        return self.base_message.calc_log_base_measure(x)

    @transform
    def to_canonical_form(self, x) -> np.ndarray:
        return self.base_message.to_canonical_form(x)

    @property
    @inverse_transform
    def mean(self) -> np.ndarray:
        return self.base_message.mean

    @property
    def variance(self) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        variance = self.base_message.variance
        mean = self.base_message.mean
        for _transform in self.transforms:
            mean = _transform.inv_transform(mean)
            jacobian = _transform.jacobian(mean)
            variance = jacobian.invquad(variance)

        return variance

    @inverse_transform
    def _sample(self, n_samples) -> np.ndarray:
        return self.base_message._sample(n_samples)

    def _factor(self, _, x: Union[np.ndarray, float],) -> np.ndarray:
        log_det = self.transform_det(x)
        x = self._transform(x)
        eta = self.base_message._broadcast_natural_parameters(x)
        t = self.base_message.to_canonical_form(x)
        log_base = self.calc_log_base_measure(x) + log_det
        return self.base_message.natural_logpdf(eta, t, log_base, self.log_partition)

    def factor(self, x: Union[float, np.ndarray]) -> Union[np.ndarray, float]:
        """
        Call the factor. The closer to the mean a given value is the higher
        the probability returned.

        Parameters
        ----------
        x
            A value in the space of the transformed message.

        Returns
        -------
        The probability this value is correct
        """
        return self._factor(self, x)

    @property
    def multivariate(self):
        return self.base_message.multivariate

    def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        jacobians = []
        for _transform in reversed(self.transforms):
            x, jacobian = _transform.transform_jac(x)
            jacobians.append(jacobian)

        log_likelihood, gradient = self.base_message.logpdf_gradient(x)

        for jacobian in reversed(jacobians):
            gradient = gradient * jacobian

        return log_likelihood, gradient

    def from_mode(self, mode: np.ndarray, covariance: np.ndarray, **kwargs):
        jac = None
        for _transform in reversed(self.transforms):
            mode, jac = _transform.transform_jac(mode)

        if covariance.shape != ():
            covariance = jac.quad(covariance)

        return self.with_base(self.base_message.from_mode(mode, covariance, **kwargs))

    def update_invalid(self, other: "TransformedMessage") -> "MessageInterface":
        return self.with_base(self.base_message.update_invalid(other.base_message))

    @property
    def log_base_measure(self):
        return self.base_message.log_base_measure

    def zeros_like(self) -> "MessageInterface":
        return self ** 0.