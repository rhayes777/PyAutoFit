from typing import Optional

from autofit.jax_wrapper import register_pytree_node_class

from autofit.messages.normal import NormalMessage
from .abstract import Prior


@register_pytree_node_class
class GaussianPrior(Prior):
    __identifier_fields__ = ("lower_limit", "upper_limit", "mean", "sigma")
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
        A prior with a uniform distribution, defined between a lower limit and upper limit.

        The conversion of an input unit value, ``u``, to a physical value, ``p``, via the prior is as follows:

        .. math::

            p = \mu + (\sigma * sqrt(2) * erfcinv(2.0 * (1.0 - u))

        For example for ``prior = GaussianPrior(mean=1.0, sigma=2.0)``, an
        input ``prior.value_for(unit=0.5)`` is equal to 1.0.

        The mapping is performed using the message functionality, where a message represents the distirubtion
        of this prior.

        Parameters
        ----------
        mean
            The mean of the Gaussian distribution defining the prior.
        sigma
            The sigma value of the Gaussian distribution defining the prior.
        lower_limit
            A lower limit of the Gaussian distribution; physical values below this value are rejected.
        upper_limit
            A upper limit of the Gaussian distribution; physical values below this value are rejected.

        Examples
        --------

        prior = af.GaussianPrior(mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=2.0)

        physical_value = prior.value_for(unit=0.5)
        """
        super().__init__(
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            message=NormalMessage(
                mean=mean,
                sigma=sigma,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            ),
            id_=id_,
        )

    def tree_flatten(self):
        return (self.mean, self.sigma, self.lower_limit, self.upper_limit), (self.id,)

    @classmethod
    def with_limits(cls, lower_limit: float, upper_limit: float) -> "GaussianPrior":
        """
        Create a new gaussian prior centred between two limits
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
        A new GaussianPrior
        """
        return cls(
            mean=(lower_limit + upper_limit) / 2,
            sigma=upper_limit - lower_limit,
        )

    def dict(self) -> dict:
        """
        A dictionary representation of this prior
        """
        prior_dict = super().dict()
        return {**prior_dict, "mean": self.mean, "sigma": self.sigma}

    @property
    def parameter_string(self) -> str:
        return f"mean = {self.mean}, sigma = {self.sigma}"
