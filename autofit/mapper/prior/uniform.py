from autofit.jax_wrapper import register_pytree_node_class
from typing import Optional

from autofit.messages.normal import UniformNormalMessage
from .abstract import Prior
from .abstract import epsilon
from ...messages.composed_transform import TransformedMessage
from ...messages.transform import LinearShiftTransform


@register_pytree_node_class
class UniformPrior(Prior):
    __identifier_fields__ = ("lower_limit", "upper_limit")

    def __init__(
        self,
        lower_limit: float = 0.0,
        upper_limit: float = 1.0,
        id_: Optional[int] = None,
    ):
        """
        A prior with a uniform distribution, defined between a lower limit and upper limit.

        The conversion of an input unit value, ``u``, to a physical value, ``p``, via the prior is as follows:

        .. math::

        For example for ``prior = UniformPrior(lower_limit=0.0, upper_limit=2.0)``, an
        input ``prior.value_for(unit=0.5)`` is equal to 1.0.

        [Rich describe how this is done via message]

        Parameters
        ----------
        lower_limit
            The lower limit of the uniform distribution defining the prior.
        upper_limit
            The upper limit of the uniform distribution defining the prior.

        Examples
        --------

        prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.2)
        """

        lower_limit = float(lower_limit)
        upper_limit = float(upper_limit)

        message = TransformedMessage(
            UniformNormalMessage,
            LinearShiftTransform(shift=lower_limit, scale=upper_limit - lower_limit),
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )
        super().__init__(
            message,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
        )

    def tree_flatten(self):
        return (self.lower_limit, self.upper_limit), (self.id,)

    def logpdf(self, x):
        # TODO: handle x as a numpy array
        if x == self.lower_limit:
            x += epsilon
        elif x == self.upper_limit:
            x -= epsilon
        return self.message.logpdf(x)

    @property
    def parameter_string(self) -> str:
        return f"lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}"

    def value_for(self, unit: float, ignore_prior_limits: bool = False) -> float:
        """
        Returns a physical value from an input unit value according to the limits of the uniform prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.

        Returns
        -------
        value
            The unit value mapped to a physical value according to the prior.

        Examples
        --------

        prior = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.2)
        """
        return round(
            super().value_for(unit, ignore_prior_limits=ignore_prior_limits), 14
        )

    # noinspection PyUnusedLocal
    @staticmethod
    def log_prior_from_value(value):
        """
        Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.

        This is used by certain non-linear searches (e.g. Emcee) in the log likelihood function evaluation.

        For a UniformPrior this is always zero, provided the value is between the lower and upper limit.
        """
        return 0.0
