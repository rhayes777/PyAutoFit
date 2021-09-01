import numpy as np

from autoconf import conf
from autofit import exc
from autofit.messages.normal import NormalMessage, UniformNormalMessage
from autofit.messages.transform import log_10_transform
from autofit.messages.transform_wrapper import TransformedWrapper


class Limits:
    @staticmethod
    def for_class_and_attributes_name(cls, attribute_name):
        limit_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name, "gaussian_limits"]
        )
        return limit_dict["lower"], limit_dict["upper"]


class UniformWrapper(
    TransformedWrapper
):
    __identifier_fields__ = ("lower_limit", "upper_limit")

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"UniformPrior, lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"

    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1
        Returns
        -------
        value: Float
            A value for the attribute between the upper and lower limits
        """
        return round(super().value_for(unit), 14)

    # noinspection PyUnusedLocal
    @staticmethod
    def log_prior_from_value(value):
        """
        Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.

        This is used by Emcee in the log likelihood function evaluation.

        NOTE: For a UniformPrior this is always zero, provided the value is between the lower and upper limit. Given
        this is check for when the instance is made (in the *instance_from_vector* function), we thus can simply return
        zero in this function.
        """
        return 0.0


class UniformPrior:
    """A prior with a uniform distribution between a lower and upper limit"""

    def __new__(
            cls,
            lower_limit=0.0,
            upper_limit=1.0,
            log_norm=0.0,
            id_=None
    ):
        UniformPrior = UniformNormalMessage.shifted(
            shift=lower_limit,
            scale=(upper_limit - lower_limit),
            wrapper_cls=UniformWrapper
        )

        UniformPrior.__class_path__ = cls
        return UniformPrior(
            0.0,
            1.0,
            id_=id_,
            lower_limit=float(lower_limit),
            upper_limit=float(upper_limit),
        )


class LogUniformWrapper(
    TransformedWrapper
):
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

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"LogUniformPrior, lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"


class LogUniformPrior:
    """A prior with a uniform distribution between a lower and upper limit"""

    def __new__(
            cls,
            lower_limit=1e-6,
            upper_limit=1.0,
            log_norm=0.0,
            id_=None
    ):
        if lower_limit <= 0.0:
            raise exc.PriorException(
                "The lower limit of a LogUniformPrior cannot be zero or negative."
            )

        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        LogUniformPrior = UniformNormalMessage.shifted(
            shift=np.log10(lower_limit),
            scale=np.log10(upper_limit / lower_limit),
        ).transformed(
            log_10_transform,
            wrapper_cls=LogUniformWrapper
        )

        LogUniformPrior.__class_path__ = cls
        return LogUniformPrior(
            0.0,
            1.0,
            id_=id_,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )


class GaussianPrior(NormalMessage):
    """A prior with a gaussian distribution"""
