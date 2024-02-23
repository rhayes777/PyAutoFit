from typing import Optional

import numpy as np

from autofit.messages.normal import NormalMessage
from .abstract import Prior
from ...messages.composed_transform import TransformedMessage
from ...messages.transform import log_transform


class LogGaussianPrior(Prior):
    __identifier_fields__ = ("lower_limit", "upper_limit", "mean", "sigma")
    __database_args__ = ("mean", "sigma", "lower_limit", "upper_limit", "id_")

    def __init__(
        self,
        mean: float,
        sigma: float,
        lower_limit: float = 0.0,
        upper_limit: float = float("inf"),
        id_: Optional[int] = None,
    ):
        """
        A prior for a variable whose logarithm is gaussian distributed. Work in natural log.

        The conversion of an input unit value, ``u``, to a physical value, ``p``, via the prior is as follows:

        .. math::

            p = \mu + (\sigma * sqrt(2) * erfcinv(2.0 * (1.0 - u))

        For example for ``prior = LogGaussianPrior(mean=1.0, sigma=2.0)``, an
        input ``prior.value_for(unit=0.5)`` is equal to 1.0.

        [Rich describe how this is done via message]

        Parameters
        ----------
        mean
            The *natural log* of the distribution's mean.
        sigma
            The spread of this distribution in *natural log* space, e.g. sigma=1.0 means P(ln x) has a
            standard deviation of 1.
        lower_limit
            A lower limit in *real space* (not log); physical values below this are rejected.
        upper_limit
            A upper limit in *real space* (not log); physical values above this are rejected.

        Examples
        --------

        prior = af.LogGaussianPrior(mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.5)
        """
        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        self.mean = mean
        self.sigma = sigma

        message = TransformedMessage(
            NormalMessage(mean, sigma),
            log_transform,
        )

        super().__init__(
            message=message,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
        )

    def _new_for_base_message(self, message):
        """
        Create a new instance of this wrapper but change the parameters used
        to instantiate the underlying message. This is useful for retaining
        the same transform stack after recreating the underlying message during
        projection.
        """
        return LogGaussianPrior(
            *message.parameters,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
            id_=self.instance().id,
        )

    def value_for(self, unit: float, ignore_prior_limits: bool = False) -> float:
        """
        Return a physical value for a value between 0 and 1 with the transformation
        described by this prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.

        Returns
        -------
        A physical value, mapped from the unit value accoridng to the prior.
        """
        return super().value_for(unit, ignore_prior_limits=ignore_prior_limits)

    @property
    def parameter_string(self) -> str:
        return f"mean = {self.mean}, sigma = {self.sigma}"

    def log_prior_from_value(self, value):
        if value <= 0:
            return float("-inf")

        return self.message.base_message.log_prior_from_value(np.log(value)) - np.log(
            value
        )
