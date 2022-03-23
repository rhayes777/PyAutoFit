import numpy as np

from autoconf import conf
from autofit import exc
from autofit.messages.normal import NormalMessage, UniformNormalMessage, LogNormalMessage
from autofit.messages.transform import log_10_transform
from autofit.messages.transform_wrapper import TransformedWrapperInstance
from .abstract import epsilon, assert_within_limits


class Limits:
    @staticmethod
    def for_class_and_attributes_name(cls, attribute_name):
        limit_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name, "gaussian_limits"]
        )
        return limit_dict["lower"], limit_dict["upper"]


class WrappedInstance(
    TransformedWrapperInstance
):
    __identifier_fields__ = ("lower_limit", "upper_limit")

    __database_args__ = (
        "lower_limit",
        "upper_limit",
        "log_norm",
        "id_",
    )

    def __init__(
            self,
            transformed_wrapper,
            *args,
            lower_limit,
            upper_limit,
            **kwargs
    ):
        super().__init__(
            transformed_wrapper,
            *args,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            **kwargs
        )
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        if self.lower_limit >= self.upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )

    def _new_for_base_message(
            self,
            message
    ):
        return type(self)(
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
            id_=self.instance().id,
            params=message.parameters
        )


class UniformPrior(WrappedInstance):
    """A prior with a uniform distribution between a lower and upper limit"""

    def __init__(
            self,
            lower_limit=0.0,
            upper_limit=1.0,
            id_=None,
            params=(0.0, 1.0)
    ):
        if any(map(np.isnan, params)):
            raise exc.MessageException(
                "nan parameter passed to UniformPrior"
            )
        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        Message = UniformNormalMessage.shifted(
            shift=lower_limit,
            scale=upper_limit - lower_limit,
        )
        super().__init__(
            Message,
            *params,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_
        )

    def logpdf(self, x):
        # TODO: handle x as a numpy array
        if x == self.lower_limit:
            x += epsilon
        elif x == self.upper_limit:
            x -= epsilon
        return self.instance().logpdf(x)

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"UniformPrior, lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"

    @assert_within_limits
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


class LogUniformPrior(WrappedInstance):
    """A prior with a uniform distribution between a lower and upper limit"""

    def __init__(
            cls,
            lower_limit=1e-6,
            upper_limit=1.0,
            id_=None,
            params=(0.0, 1.0)
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
            Message,
            *params,
            id_=id_,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
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

    @assert_within_limits
    def value_for(self, unit: float) -> float:
        return super().value_for(unit)

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"LogUniformPrior, lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"


class GaussianPrior(NormalMessage):
    """A prior with a gaussian distribution"""

    __identifier_fields__ = (
        "lower_limit",
        "upper_limit",
        "mean",
        "sigma"
    )

    @classmethod
    def with_limits(
            cls,
            lower_limit: float,
            upper_limit: float
    ) -> "GaussianPrior":
        """
        Create a new gaussian prior centred between two limits
        with sigma distance between this limits.

        Note that these limits are not strict so exceptions will not
        be raised for values outside of the limits.

        Parameters
        ----------
        lower_limit
        upper_limit

        Returns
        -------
        A new GaussianPrior
        """
        return cls(
            mean=(lower_limit + upper_limit) / 2,
            sigma=upper_limit - lower_limit,
        )

    @assert_within_limits
    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1

        Returns
        -------
        value: Float
            A value for the attribute biased to the gaussian distribution
        """
        return super().value_for(unit)


class LogGaussianPrior(WrappedInstance):
    """A prior with a log gaussian distribution"""

    __identifier_fields__ = (
        "lower_limit",
        "upper_limit",
        "mean",
        "sigma"
    )

    def __init__(
            self,
            mean,
            sigma,
            lower_limit=0.0,
            upper_limit=float("inf"),
            id_=None,
    ):
        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        self.mean = mean
        self.sigma = sigma

        super().__init__(
            LogNormalMessage,
            mean=mean,
            sigma=sigma,
            id_=id_,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

    def _new_for_base_message(
            self,
            message
    ):
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
            id_=self.instance().id
        )

    @assert_within_limits
    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1

        Returns
        -------
        value: Float
            A value for the attribute biased to the gaussian distribution
        """
        return super().value_for(unit)
