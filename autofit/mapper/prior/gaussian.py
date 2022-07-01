from autofit.messages.normal import NormalMessage
from .abstract import Prior


class GaussianPrior(Prior):
    """A prior with a gaussian distribution"""

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
            lower_limit=float("-inf"),
            upper_limit=float("inf"),

    ):
        super().__init__(
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            message=NormalMessage(
                mean=mean,
                sigma=sigma,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
            )
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

    def dict(self) -> dict:
        """
        A dictionary representation of this prior
        """
        prior_dict = super().dict()
        return {**prior_dict, "mean": self.mean, "sigma": self.sigma}

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"GaussianPrior, mean = {self.mean}, sigma = {self.sigma}"
