import numpy as np
from scipy.stats import norm

from autofit.mapper.prior.gaussian import GaussianPrior
from autofit.mapper.prior.truncated_gaussian import TruncatedGaussianPrior
from autofit.mapper.prior.uniform import UniformPrior
from autofit.mapper.prior.log_uniform import LogUniformPrior

class PriorVectorized:

    def __init__(self, model):

        self.model = model
        self.prior_list = model.priors_ordered_by_id

        supported_prior_list = [
            UniformPrior,
            GaussianPrior,
            TruncatedGaussianPrior,
            LogUniformPrior,
        ]

        # 1) Group UniformPriors with key info
        self.uniform_idx = [
            i for i, p in enumerate(self.prior_list) if isinstance(p, UniformPrior)
        ]

        if self.uniform_idx:

            self.uniform_lowers = np.array(
                [self.prior_list[i].lower_limit for i in self.uniform_idx]
            )  # (n_uniforms,)
            self.uniform_uppers = np.array(
                [self.prior_list[i].upper_limit for i in self.uniform_idx]
            )  # (n_uniforms,)

        self.gaussian_idx = [
            i for i, p in enumerate(self.prior_list) if isinstance(p, GaussianPrior)
        ]

        if self.gaussian_idx:

            self.gaussian_means = np.array(
                [self.prior_list[i].mean for i in self.gaussian_idx]
            )
            self.gaussian_sigmas = np.array(
                [self.prior_list[i].sigma for i in self.gaussian_idx]
            )

        self.truncated_gaussian_idx = [
            i
            for i, p in enumerate(self.prior_list)
            if isinstance(p, TruncatedGaussianPrior)
        ]

        if self.truncated_gaussian_idx:

            self.truncated_gaussian_means = np.array(
                [self.prior_list[i].mean for i in self.truncated_gaussian_idx]
            )
            self.truncated_gaussian_sigmas = np.array(
                [self.prior_list[i].sigma for i in self.truncated_gaussian_idx]
            )
            lowers = np.array(
                [self.prior_list[i].lower_limit for i in self.truncated_gaussian_idx]
            )
            uppers = np.array(
                [self.prior_list[i].upper_limit for i in self.truncated_gaussian_idx]
            )

            a = (
                        lowers - self.truncated_gaussian_means
                ) / self.truncated_gaussian_sigmas
            b = (
                        uppers - self.truncated_gaussian_means
                ) / self.truncated_gaussian_sigmas

            self.truncated_gaussian_cdf_a = norm.cdf(a)
            self.truncated_gaussian_cdf_b = norm.cdf(b)

        self.loguniform_idx = [
            i
            for i, p in enumerate(self.prior_list)
            if isinstance(p, LogUniformPrior)
        ]

        if self.loguniform_idx:

            self.loguniform_lowers = np.array(
                [self.prior_list[i].lower_limit for i in self.loguniform_idx]
            )  # (n_loguniforms,)
            self.loguniform_uppers = np.array(
                [self.prior_list[i].upper_limit for i in self.loguniform_idx]
            )  # (n_loguniforms,)
            # Map unit interval to log scale:
            # x = exp(log(lower) + unit * (log(upper) - log(lower)))
            self.loguniform_log_lowers = np.log(self.loguniform_lowers)
            self.loguniform_log_uppers = np.log(self.loguniform_uppers)

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        """
        Vectorized transform of a (n_samples, n_priors) `cube` array
        by grouping priors by type and processing each group in one go.
        """

        n_samples, n_priors = cube.shape
        out = np.empty_like(cube)

        # 2) Batch‐process all UniformPriors
        if self.uniform_idx:
            subcube = cube[:, self.uniform_idx]  # shape (n_samples, n_uniforms)

            out[:, self.uniform_idx] = self.uniform_lowers + subcube * (
                    self.uniform_uppers - self.uniform_lowers
            )

        # 3) Batch‐process all GaussianPriors
        if self.gaussian_idx:
            subcube = cube[:, self.gaussian_idx]  # (n_samples, n_gaussians)

            inv = norm.ppf(subcube)  # inverse CDF of standard normal
            out[:, self.gaussian_idx] = self.gaussian_means + self.gaussian_sigmas * inv

        # 4) Batch‐process all TruncatedGaussianPriors
        if self.truncated_gaussian_idx:

            subcube = cube[:, self.truncated_gaussian_idx]  # (n_samples, n_truncs)

            truncated_cdf = self.truncated_gaussian_cdf_a + subcube * (
                    self.truncated_gaussian_cdf_b - self.truncated_gaussian_cdf_a
            )
            x_std = norm.ppf(truncated_cdf)

            out[:, self.truncated_gaussian_idx] = (
                    self.truncated_gaussian_means + self.truncated_gaussian_sigmas * x_std
            )

        # 5) Batch‐process all LogUniformPriors
        if self.loguniform_idx:

            subcube = cube[:, self.loguniform_idx]  # (n_samples, n_loguniforms)

            out[:, self.loguniform_idx] = np.exp(
                self.loguniform_log_lowers
                + subcube * (self.loguniform_log_uppers - self.loguniform_log_lowers)
            )

        return out