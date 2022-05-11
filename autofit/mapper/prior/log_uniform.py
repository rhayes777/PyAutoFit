import numpy as np

from autofit import exc
from autofit.messages.normal import UniformNormalMessage
from autofit.messages.transform import log_10_transform
from .abstract import Prior
from .wrapped_instance import WrappedInstance


class LogUniformPrior(Prior):
    """A prior with a uniform distribution between a lower and upper limit"""

    def __init__(
            self,
            lower_limit=1e-6,
            upper_limit=1.0,
            id_=None,
    ):
        if lower_limit <= 0.0:
            raise exc.PriorException(
                "The lower limit of a LogUniformPrior cannot be zero or negative."
            )

        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        Message = UniformNormalMessage.shifted(
            shift=np.log10(lower_limit),
            scale=np.log10(upper_limit / lower_limit),
        ).transformed(
            log_10_transform
        )

        super().__init__(
            message=WrappedInstance(
                Message,
                0.0, 1.0,
                id_=id_,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            ),
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
        )

    @classmethod
    def with_limits(
            cls,
            lower_limit: float,
            upper_limit: float
    ):
        return cls(
            lower_limit=max(
                0.000001,
                lower_limit
            ),
            upper_limit=upper_limit,
        )

    __identifier_fields__ = ("lower_limit", "upper_limit")

    @staticmethod
    def log_prior_from_value(value):
        """
        Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
            posterior as log_prior + log_likelihood.

        This is used by Emcee in the log likelihood function evaluation.

        Parameters
        ----------
        value : float
            The physical value of this prior's corresponding parameter in a `NonLinearSearch` sample."""
        return 1.0 / value

    def value_for(self, unit: float, ignore_prior_limits=False) -> float:
        return super().value_for(unit, ignore_prior_limits=ignore_prior_limits)

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"LogUniformPrior, lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"
