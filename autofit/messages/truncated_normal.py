from collections.abc import Hashable
import math
from scipy.stats import truncnorm
from typing import Optional, Tuple, Union

import numpy as np
from autofit.jax_wrapper import erfinv
from scipy.stats import norm

from autoconf import cached_property
from autofit.mapper.operator import LinearOperator
from autofit.messages.abstract import AbstractMessage
from .composed_transform import TransformedMessage
from .transform import (
    phi_transform,
    log_transform,
    multinomial_logit_transform,
    log_10_transform,
)
from .. import exc


def is_nan(value):
    is_nan_ = np.isnan(value)
    if isinstance(is_nan_, np.ndarray):
        is_nan_ = is_nan_.all()
    return is_nan_


class TruncatedNormalMessage(AbstractMessage):
    @cached_property
    def log_partition(self) -> float:
        """
        Compute the log-partition function (normalizer) of the truncated Gaussian.

        This is the log of the normalization constant Z of the truncated normal:

            Z = Φ((b - μ)/σ) - Φ((a - μ)/σ)

        where Φ is the standard normal CDF and [a, b] are the truncation bounds.

        Returns
        -------
        float
            The log-partition (log of the normalizing constant).
        """
        a = (self.lower_limit - self.mean) / self.sigma
        b = (self.upper_limit - self.mean) / self.sigma
        Z = norm.cdf(b) - norm.cdf(a)
        return np.log(Z) if Z > 0 else -np.inf

    log_base_measure = -0.5 * np.log(2 * np.pi)

    @property
    def _support(self):
        return ((self.lower_limit, self.upper_limit),)

    _parameter_support = ((-np.inf, np.inf), (0, np.inf))

    def __init__(
        self,
        mean : Union[float, np.ndarray],
        sigma : Union[float, np.ndarray],
        lower_limit=-math.inf,
        upper_limit=math.inf,
        log_norm : Optional[float] = 0.0,
        id_ : Optional[Hashable] = None,
    ):
        """
        A Gaussian (Normal) message representing a probability distribution over a continuous variable.

        This message defines a Normal distribution parameterized by its mean and standard deviation (sigma).

        Parameters
        ----------
        mean
            The mean (μ) of the normal distribution.

        sigma
            The standard deviation (σ) of the distribution. Must be non-negative.

        log_norm
            An additive constant to the log probability of the message. Used internally for message-passing normalization.
            Default is 0.0.

        id_
            An optional unique identifier used to track the message in larger probabilistic graphs or models.
        """
        if (np.array(sigma) < 0).any():
            raise exc.MessageException("Sigma cannot be negative")

        super().__init__(
            mean,
            sigma,
            float(lower_limit),
            float(upper_limit),
            log_norm=log_norm,
            id_=id_,
        )
        self.mean, self.sigma, self.lower_limit, self.upper_limit = self.parameters

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the cumulative distribution function (CDF) of the truncated Gaussian distribution
        at a given value or array of values `x`.

        The CDF is computed using `scipy.stats.truncnorm`, which handles the normalization
        over the truncated interval [lower_limit, upper_limit].

        Parameters
        ----------
        x
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        The cumulative probability P(X ≤ x) under the truncated Gaussian.
        """
        a = (self.lower_limit - self.mean) / self.sigma
        b = (self.upper_limit - self.mean) / self.sigma
        return truncnorm.cdf(x, a=a, b=b, loc=self.mean, scale=self.sigma)

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the percent-point function (inverse CDF) of the truncated Gaussian distribution.

        This function maps a probability value `x` in [0, 1] to the corresponding value
        under the truncated Gaussian distribution.

        Parameters
        ----------
        x
            The cumulative probability or array of probabilities.

        Returns
        -------
        The value(s) corresponding to the input quantiles.
        """
        a = (self.lower_limit - self.mean) / self.sigma
        b = (self.upper_limit - self.mean) / self.sigma
        return truncnorm.ppf(x, a=a, b=b, loc=self.mean, scale=self.sigma)

    @cached_property
    def natural_parameters(self) -> np.ndarray:
        """
        The pseudo-natural (canonical) parameters of a truncated Gaussian distribution.

        For a Gaussian with mean μ and standard deviation σ, the untruncated natural parameters η are:

            η₁ = μ / σ²
            η₂ = -1 / (2σ²)

        These are returned here even for the truncated case, but note that due to truncation,
        the distribution is no longer in the exponential family and the log-partition function
        depends on the lower and upper truncation limits.

        Returns
        -------
        A NumPy array containing the pseudo-natural parameters [η₁, η₂].
        """
        return self.calc_natural_parameters(self.mean, self.sigma)

    @staticmethod
    def calc_natural_parameters(mu : Union[float, np.ndarray], sigma : Union[float, np.ndarray]) -> np.ndarray:
        """
        Convert standard parameters of a Gaussian distribution (mean and standard deviation)
        into natural parameters used in its exponential family representation.

        This function does **not** directly account for truncation. In the case of a truncated Gaussian,
        these parameters are treated as pseudo-natural parameters, meaning they are defined analogously
        to the untruncated case but do not fully characterize the distribution. This is because truncation
        modifies the normalization constant (log-partition function), making the distribution fall outside
        the strict exponential family.

        For truncated Gaussians, any computations involving expectations, gradients, or log-partition
        functions must incorporate the effects of truncation separately.

        Parameters
        ----------
        mu
            Mean of the Gaussian distribution.
        sigma
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        Natural parameters [η₁, η₂], where:
            η₁ = μ / σ²
            η₂ = -1 / (2σ²)
        """
        precision = 1 / sigma**2
        return np.array([mu * precision, -precision / 2])

    @staticmethod
    def invert_natural_parameters(natural_parameters : np.ndarray) -> Tuple[float, float]:
        """
        Convert natural parameters [η₁, η₂] back into standard parameters (mean and sigma)
        of a Gaussian distribution.

        For a truncated Gaussian, this inversion treats the natural parameters as if they
        came from an untruncated distribution. That is, the computed (mean, sigma) are
        the parameters of the *underlying* Gaussian prior to truncation.

        Parameters
        ----------
        natural_parameters
            The natural parameters [η₁, η₂] from the exponential family form.

        Returns
        -------
        The corresponding (mean, sigma) of the Gaussian distribution.
        """
        eta1, eta2 = natural_parameters
        mu = -0.5 * eta1 / eta2
        sigma = np.sqrt(-0.5 / eta2)
        return mu, sigma

    @staticmethod
    def to_canonical_form(x : Union[float, np.ndarray]) -> np.ndarray:
        """
        Convert a scalar input `x` to its sufficient statistics for the Gaussian exponential family.

        This form is unchanged by truncation, as sufficient statistics remain [x, x²] regardless
        of whether the distribution is truncated. However, note that for a truncated Gaussian,
        expectations (e.g. E[x], E[x²]) must be computed over the truncated support.

        Parameters
        ----------
        x
            Input data point or array of points.

        Returns
        -------
        The sufficient statistics [x, x²].
        """
        return np.array([x, x**2])

    @classmethod
    def invert_sufficient_statistics(cls, suff_stats: Tuple[float, float]) -> np.ndarray:
        """
        Convert sufficient statistics [E[x], E[x²]] into natural parameters [η₁, η₂].

        These moments are assumed to be expectations *under the truncated Gaussian* distribution,
        meaning that the inferred natural parameters correspond to the truncated form indirectly.

        Parameters
        ----------
        suff_stats
            First and second moments of the distribution.

        Returns
        -------
        Natural parameters of the Gaussian.
        """
        m1, m2 = suff_stats
        sigma = np.sqrt(m2 - m1**2)
        return cls.calc_natural_parameters(m1, sigma)

    @cached_property
    def variance(self) -> np.ndarray:
        """
        Return the variance σ² of the Gaussian distribution.
        """
        return self.sigma**2

    def sample(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Draw samples from a truncated Gaussian distribution using inverse transform sampling.

        Samples are drawn from a standard Normal distribution, transformed using the mean and sigma,
        and then rejected if they fall outside the [lower_limit, upper_limit] bounds.

        Parameters
        ----------
        n_samples
            Number of samples to draw. If None, returns a single sample.

        Returns
        -------
        Sample(s) from the truncated Gaussian distribution.
        """
        a, b = (self.lower_limit - self.mean) / self.sigma, (self.upper_limit - self.mean) / self.sigma
        shape = (n_samples,) + self.shape if n_samples else self.shape
        samples = truncnorm.rvs(a, b, loc=self.mean, scale=self.sigma, size=shape)

        return samples

    def kl(self, dist : "TruncatedNormalMessage") -> float:
        """
        Compute the Kullback-Leibler (KL) divergence between two truncated Gaussian distributions.

        This is an approximate KL divergence that assumes both distributions are truncated Gaussians
        with the same support (i.e. the same lower and upper limits). If the supports differ, this
        expression is invalid and should raise an error or be corrected for normalization.

        Parameters
        ----------
        dist
            The target distribution for the KL divergence.

        Returns
        -------
        float
            The KL divergence KL(self || dist).
        """
        if (self.lower_limit != dist.lower_limit) or (self.upper_limit != dist.upper_limit):
            raise ValueError("KL divergence between truncated Gaussians with different support is not implemented.")

        return (
            np.log(dist.sigma / self.sigma)
            + (self.sigma**2 + (self.mean - dist.mean) ** 2) / 2 / dist.sigma**2
            - 1 / 2
        )

    @classmethod
    def from_mode(
        cls,
        mode: np.ndarray,
        covariance: Union[float, LinearOperator] = 1.0,
        **kwargs
    ) -> "TruncatedNormalMessage":
        """
        Construct a truncated Gaussian from its mode and covariance.

        For a Gaussian, the mode equals the mean. This method uses that identity to construct
        the message from point estimates.

        Parameters
        ----------
        mode
            The mode (same as mean for Gaussian).
        covariance
            The covariance or a linear operator with `.diagonal()` method.
        **kwargs
            Additional keyword arguments passed to the constructor, such as truncation bounds.

        Returns
        -------
        An instance of the TruncatedNormalMessage class.
        """
        if isinstance(covariance, LinearOperator):
            variance = covariance.diagonal()
        else:
            mode, variance = cls._get_mean_variance(mode, covariance)
        return cls(mode, np.abs(variance) ** 0.5, **kwargs)

    def _normal_gradient_hessian(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def logpdf_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the gradient of the log-pdf of the Gaussian evaluated at `x`.

        Parameters
        ----------
        x
            Evaluation points.

        Returns
        -------
        Log-pdf values and gradients.
        """
        return self._normal_gradient_hessian(x)[:2]

    def logpdf_gradient_hessian(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the gradient and Hessian of the log-pdf of the Gaussian at `x`.

        Parameters
        ----------
        x
            Evaluation points.

        Returns
        -------
        Gradient and Hessian of the log-pdf.
        """
        return self._normal_gradient_hessian(x)

    __name__ = "truncated_gaussian_prior"

    __default_fields__ = ("log_norm", "id_")

    def value_for(self, unit: float) -> float:
        """
        Map a unit value in [0, 1] to a physical value drawn from this truncated Gaussian prior.

        For a truncated Gaussian, this is done using the percent-point function (inverse CDF)
        that accounts for the truncation bounds.

        Parameters
        ----------
        unit
            A unit value between 0 and 1 representing a uniform draw.

        Returns
        -------
        A physical value sampled from the truncated Gaussian prior corresponding to the given unit.

        Examples
        --------
        >>> prior = af.TruncatedNormalMessage(mean=1.0, sigma=2.0, lower_limit=0.0, upper_limit=2.0)
        >>> physical_value = prior.value_for(unit=0.5)
        """
        # Standardized truncation bounds
        a = (self.lower_limit - self.mean) / self.sigma
        b = (self.upper_limit - self.mean) / self.sigma

        # Interpolate unit into [Phi(a), Phi(b)]
        lower_cdf = norm.cdf(a)
        upper_cdf = norm.cdf(b)
        truncated_cdf = lower_cdf + unit * (upper_cdf - lower_cdf)

        # Map back to x using inverse CDF, then rescale
        x_standard = norm.ppf(truncated_cdf)
        return self.mean + self.sigma * x_standard

    def log_prior_from_value(self, value: float) -> float:
        """
        Compute the log prior probability of a given physical value under this truncated Gaussian prior.

        This accounts for truncation by normalizing the Gaussian density over the
        interval [lower_limit, upper_limit], returning -inf if the value lies outside
        these limits.

        Parameters
        ----------
        value
            A physical parameter value for which the log prior is evaluated.

        Returns
        -------
        The log prior probability of the given value, or -inf if outside truncation bounds.
        """
        # Check truncation bounds
        if not (self.lower_limit <= value <= self.upper_limit):
            return -np.inf

        # Standardized truncation limits
        a, b = (self.lower_limit - self.mean) / self.sigma, (self.upper_limit - self.mean) / self.sigma

        # Normalization constant for truncated Gaussian
        Z = norm.cdf(b) - norm.cdf(a)

        # Standardized value
        z = (value - self.mean) / self.sigma

        # Log probability density of normal (up to normalization)
        log_pdf = -0.5 * z ** 2 - np.log(self.sigma) - 0.5 * np.log(2 * np.pi)

        # Adjust for truncation normalization
        return log_pdf - np.log(Z)

    def __str__(self):
        """
        Generate a short string summary describing the prior for use in model summaries.
        """
        return (f"TruncatedNormalMessage, mean = {self.mean}, sigma = {self.sigma}, "
                f"lower_limit = {self.lower_limit}, upper_limit = {self.upper_limit}")

    def __repr__(self):
        """
        Return the official string representation of this Gaussian prior including
        the ID, mean, sigma, and optional bounds.
        """
        return (
            "<TruncatedNormalMessage id={} mean={} sigma={} "
            "lower_limit={} upper_limit={}>".format(
                self.id, self.mean, self.sigma, self.lower_limit, self.upper_limit
            )
        )

    @property
    def natural(self)-> "NaturalNormal":
        """
        Return a 'zeroed' natural parameterization of this Gaussian prior.

        Returns
        -------
        A natural form Gaussian with zeroed parameters but same configuration.
        """
        return TruncatedNaturalNormal.from_natural_parameters(
            self.natural_parameters * 0.0, **self._init_kwargs
        )

    def zeros_like(self) -> "AbstractMessage":
        """
        Return a new instance of this prior with the same structure but zeroed natural parameters.

        Useful for initializing messages in variational inference frameworks.

        Returns
        -------
        A new prior object with zeroed natural parameters.
        """
        return self.natural.zeros_like()


class TruncatedNaturalNormal(TruncatedNormalMessage):
    """
    Identical to the TruncatedNormalMessage but allows non-normalised values,
    e.g negative or infinite variances
    """

    _parameter_support = ((-np.inf, np.inf), (-np.inf, 0))

    def __init__(
        self,
        eta1 : float,
        eta2 : float,
        lower_limit=-math.inf,
        upper_limit=math.inf,
        log_norm : Optional[float] = 0.0,
        id_ : Optional[Hashable] = None,
    ):
        """
        A natural parameterization of a Gaussian distribution.

        This class behaves like `TruncatedNormalMessage`, but allows non-normalized or degenerate distributions,
        including those with negative or infinite variance. This flexibility is useful in advanced
        inference settings like message passing or variational approximations, where intermediate
        natural parameter values may fall outside standard constraints.

        In natural form, the parameters `eta1` and `eta2` correspond to:
            - eta1 = mu / sigma^2
            - eta2 = -1 / (2 * sigma^2)

        Parameters
        ----------
        eta1
            First natural parameter, related to the mean.
        eta2
            Second natural parameter, related to the variance (must be < 0).
        log_norm
            Optional additive normalization term for use in message passing.
        id_
            Optional identifier for the distribution instance.
        """
        AbstractMessage.__init__(
            self,
            eta1,
            eta2,
            log_norm=log_norm,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            id_=id_,
        )

    @cached_property
    def sigma(self) -> float:
        """
        Return the standard deviation σ of the truncated Gaussian corresponding to
        the natural parameters and truncation limits.

        Uses scipy.stats.truncnorm to compute std dev on the truncated interval.

        Returns
        -------
        The truncated Gaussian standard deviation σ.
        """
        precision = -2 * self.parameters[1]
        if precision <= 0 or np.isinf(precision) or np.isnan(precision):
            # Degenerate or invalid precision: fallback to NaN or zero
            return np.nan

        mean = -self.parameters[0] / (2 * self.parameters[1])
        std = precision ** -0.5

        a, b = (self.lower_limit - mean) / std, (self.upper_limit - mean) / std

        # Compute truncated std dev
        truncated_std = truncnorm.std(a, b, loc=mean, scale=std)
        return truncated_std

    @cached_property
    def mean(self) -> float:
        """
        Return the mean μ of the truncated Gaussian corresponding to the natural parameters
        and truncation limits.

        Uses scipy.stats.truncnorm to compute mean on the truncated interval.

        Returns
        -------
        The truncated Gaussian mean μ.
        """
        precision = -2 * self.parameters[1]
        if precision <= 0 or np.isinf(precision) or np.isnan(precision):
            # Degenerate or invalid precision: fallback to NaN or zero
            return np.nan

        mean = -self.parameters[0] / (2 * self.parameters[1])
        std = precision**-0.5

        a, b = (self.lower_limit - mean) / std, (self.upper_limit - mean) / std

        # Compute truncated mean
        truncated_mean = truncnorm.mean(a, b, loc=mean, scale=std)
        return truncated_mean

    @staticmethod
    def calc_natural_parameters(
            eta1: float,
            eta2: float,
            lower_limit: float = -np.inf,
            upper_limit: float = np.inf
    ) -> np.ndarray:
        """
        Return the natural parameters in array form (identity function for this class).

        Currently returns eta1 and eta2 ignoring truncation,
        but can be extended to adjust natural parameters based on truncation.

        Parameters
        ----------
        eta1
            The first natural parameter.
        eta2
            The second natural parameter.
        """
        return np.array([eta1, eta2])

    @cached_property
    def natural_parameters(self) -> np.ndarray:
        """
        Return the natural parameters of this distribution.
        """
        return self.calc_natural_parameters(*self.parameters, self.lower_limit, self.upper_limit)

    @classmethod
    def invert_sufficient_statistics(
            cls,
            suff_stats: Tuple[float, float],
            lower_limit: float = -np.inf,
            upper_limit: float = np.inf
    ) -> np.ndarray:
        """
        Convert sufficient statistics back to natural parameters.

        Parameters
        ----------
        suff_stats
            Tuple of first and second moments: (mean, second_moment).

        Returns
        -------
        Natural parameters [eta1, eta2] recovered from the sufficient statistics.
        """
        m1, m2 = suff_stats
        precision = 1 / (m2 - m1**2)
        return cls.calc_natural_parameters(m1 * precision, -precision / 2, lower_limit, upper_limit)

    @staticmethod
    def invert_natural_parameters(natural_parameters: np.ndarray) -> np.ndarray:
        """
        Identity function for natural parameters (no inversion needed).

        Parameters
        ----------
        natural_parameters : np.ndarray
            Natural parameters [eta1, eta2].

        Returns
        -------
        np.ndarray
            The same input array.
        """
        return natural_parameters

    @classmethod
    def from_mode(
            cls,
            mode: np.ndarray,
            covariance: Union[float, LinearOperator] = 1.0,
            lower_limit: float = -np.inf,
            upper_limit: float = np.inf,
            **kwargs
    ) -> "NaturalNormal":
        """
        Construct a `NaturalNormal` distribution from mode and covariance.

        Parameters
        ----------
        mode
            The mode (mean) of the distribution.
        covariance
            Covariance of the distribution. If a `LinearOperator`, its inverse is used for precision.
        kwargs
            Additional keyword arguments passed to the constructor.

        Returns
        -------
        An instance of `NaturalNormal` with the corresponding natural parameters.
        """
        if isinstance(covariance, LinearOperator):
            precision = covariance.inv().diagonal()
        else:
            mode, variance = cls._get_mean_variance(mode, covariance)
            precision = 1 / variance

        return cls(mode * precision, -precision / 2, lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)

    zeros_like = AbstractMessage.zeros_like

    @property
    def natural(self) -> "NaturalNormal":
        """
        Return self — already in natural form -- for clean API.
        """
        return self


UniformNormalMessage = TransformedMessage(TruncatedNormalMessage(0, 1), phi_transform)

Log10UniformNormalMessage = TransformedMessage(UniformNormalMessage, log_10_transform)

LogNormalMessage = TransformedMessage(TruncatedNormalMessage(0, 1), log_transform)
Log10NormalMessage = TransformedMessage(TruncatedNormalMessage(0, 1), log_10_transform)

# Support is the simplex
MultiLogitNormalMessage = TransformedMessage(
    TruncatedNormalMessage(0, 1), multinomial_logit_transform
)
