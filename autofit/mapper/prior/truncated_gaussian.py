from typing import Optional, Tuple

from autofit.jax_wrapper import register_pytree_node_class

from autofit.messages.truncated_normal import TruncatedNormalMessage
from .abstract import Prior


@register_pytree_node_class
class TruncatedGaussianPrior(Prior):
    __identifier_fields__ = ("mean", "sigma", "lower_limit", "upper_limit")
    __database_args__ = ("mean", "sigma", "lower_limit", "upper_limit", "id_")

    def __init__(
        self,
        mean: float,
        sigma: float,
        lower_limit: float = float("-inf"),
        upper_limit: float = float("inf"),
        id_: Optional[int] = None,
    ):
        """
        A Gaussian prior defined by a normal distribution with optional truncation limits.

        This prior represents a Gaussian (normal) distribution with mean `mean`
        and standard deviation `sigma`, optionally truncated between `lower_limit`
        and `upper_limit`. The transformation from a unit interval input `u ∈ [0, 1]`
        to a physical parameter value `p` uses the inverse error function (erfcinv) applied
        to the Gaussian CDF, adjusted for truncation:

        .. math::
            p = \mu + \sigma \sqrt{2} \, \mathrm{erfcinv}(2 \times (1 - u))

        where :math:`\mu` is the mean and :math:`\sigma` the standard deviation.

        If truncation limits are specified, values outside the interval
        [`lower_limit`, `upper_limit`] are disallowed and the distribution is
        normalized over this interval.

        Parameters
        ----------
        mean
            The mean (center) of the Gaussian prior distribution.
        sigma
            The standard deviation (spread) of the Gaussian prior.
        lower_limit : float, optional
            The lower truncation limit (default: -∞).
        upper_limit : float, optional
            The upper truncation limit (default: +∞).
        id_ : Optional[int], optional
            Optional identifier for the prior instance.

        Examples
        --------
        Create a TruncatedGaussianPrior with mean 1.0, sigma 2.0, truncated between 0.0 and 2.0:

        >>> prior = TruncatedGaussianPrior(mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=2.0)
        >>> physical_value = prior.value_for(unit=0.5)  # Returns a value near 1.0 (mean)
        """
        super().__init__(
            message=TruncatedNormalMessage(
                mean=mean,
                sigma=sigma,
                lower_limit=float(lower_limit),
                upper_limit=float(upper_limit),
            ),
            id_=id_,
        )

    def tree_flatten(self):
        return (self.mean, self.sigma, self.lower_limit, self.upper_limit, self.id), ()

    @classmethod
    def with_limits(cls, lower_limit: float, upper_limit: float) -> "TruncatedGaussianPrior":
        """
        Create a new truncated gaussian prior centred between two limits
        with sigma distance between this limits.

        Note that these limits are not strict so exceptions will not
        be raised for values outside of the limits.

        This function is typically used in prior passing, where the
        result of a model-fit are used to create new Gaussian priors
        centred on the previously estimated median PDF model.

        Parameters
        ----------
        lower_limit
            The lower limit of the new Gaussian prior.
        upper_limit
            The upper limit of the new Gaussian Prior.

        Returns
        -------
        A new prior instance centered between the limits.
        """
        return cls(
            mean=(lower_limit + upper_limit) / 2,
            sigma=(upper_limit - lower_limit),
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )

    def dict(self) -> dict:
        """
        Return a dictionary representation of this GaussianPrior instance,
        including mean and sigma.

        Returns
        -------
        Dictionary containing prior parameters.
        """
        prior_dict = super().dict()
        return {
            **prior_dict,
            "mean": self.mean,
            "sigma": self.sigma,
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit
        }

    @property
    def limits(self) -> Tuple[float, float]:
        return self.lower_limit, self.upper_limit

    @property
    def parameter_string(self) -> str:
        """
        Return a human-readable string summarizing the GaussianPrior parameters.
        """
        return (f"mean = {self.mean}, "
                f"sigma = {self.sigma}, "
                f"lower_limit = {self.lower_limit}, "
                f"upper_limit = {self.upper_limit}"
                )
