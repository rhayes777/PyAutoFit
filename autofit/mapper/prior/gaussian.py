import math

from scipy.special import erfcinv

from autofit.messages.normal import NormalMessage


class GaussianPrior(NormalMessage):
    """A prior with a gaussian distribution"""

    __name__ = "gaussian_prior"

    def __init__(
            self,
            mean,
            sigma,
            lower_limit=-math.inf,
            upper_limit=math.inf,
            log_norm=0.0,
            id_=None,
    ):
        super().__init__(
            mean=mean,
            sigma=sigma,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
            log_norm=log_norm
        )
        self.mu = float(mean)
        self.sigma = float(sigma)

        self._log_pdf = None

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
        return self.mean + (self.sigma * math.sqrt(2) * erfcinv(2.0 * (1.0 - unit)))

    def __call__(self, x):
        return self.logpdf(x)

    def log_prior_from_value(self, value):
        """
        Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.

        This is used by Emcee in the log likelihood function evaluation.

        Parameters
        ----------
        value : float
            The physical value of this prior's corresponding parameter in a `NonLinearSearch` sample."""
        return (value - self.mean) ** 2.0 / (2 * self.sigma ** 2.0)

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return (
            f"GaussianPrior, mean = {self.mean}, sigma = {self.sigma}"
        )

    def __repr__(self):
        return (
            "<GaussianPrior id={} mean={} sigma={} "
            "lower_limit={} upper_limit={}>".format(
                self.id, self.mean, self.sigma, self.lower_limit, self.upper_limit
            )
        )

    def dict(self) -> dict:
        """
        A dictionary representation of this prior
        """
        prior_dict = super().dict()
        return {**prior_dict, "mean": self.mean, "sigma": self.sigma}
