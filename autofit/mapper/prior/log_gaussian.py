from autofit.messages.normal import NormalMessage
from .abstract import Prior
from ...messages.composed_transform import TransformedMessage
from ...messages.transform import log_transform


class LogGaussianPrior(Prior):
    """A prior with a log gaussian distribution"""

    __identifier_fields__ = ("lower_limit", "upper_limit", "mean", "sigma")

    def __init__(
        self, mean, sigma, lower_limit=0.0, upper_limit=float("inf"), id_=None,
    ):
        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        self.mean = mean
        self.sigma = sigma

        message = TransformedMessage(NormalMessage(mean, sigma), log_transform,)

        super().__init__(
            message=message, lower_limit=lower_limit, upper_limit=upper_limit, id_=id_,
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

    def value_for(self, unit, ignore_prior_limits=False):
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
        return super().value_for(unit, ignore_prior_limits=ignore_prior_limits)

    @property
    def parameter_string(self):
        return f"mean = {self.mean}, sigma = {self.sigma}"
