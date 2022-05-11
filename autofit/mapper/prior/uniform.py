from autofit.messages.normal import UniformNormalMessage
from .abstract import Prior
from .abstract import epsilon
from .wrapped_instance import WrappedInstance


class UniformPrior(Prior):
    """A prior with a uniform distribution between a lower and upper limit"""

    def __init__(
            self,
            lower_limit=0.0,
            upper_limit=1.0,
            id_=None,
    ):
        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        Message = UniformNormalMessage.shifted(
            shift=lower_limit,
            scale=upper_limit - lower_limit,
        )
        super().__init__(
            WrappedInstance(
                Message,
                0, 1,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                id_=id_,
            ),
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
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

    def value_for(self, unit, ignore_prior_limits=False):
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
        return round(super().value_for(unit, ignore_prior_limits=ignore_prior_limits), 14)

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
