import math
from typing import List, Optional

import numpy as np

from autofit.mapper.model_mapper import ModelMapper
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings
from autofit.non_linear.samples.pdf import PDFSamples
from .optimizer import OptimizerSamples
from .sample import Sample, load_from_table


class MCMCSamples(PDFSamples):

    def __init__(
            self,
            model: ModelMapper,
            sample_list: List[Sample],
            auto_correlation_settings: AutoCorrelationsSettings,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):

        self.auto_correlation_settings = auto_correlation_settings

        super().__init__(
            model=model,
            sample_list=sample_list,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

    @property
    def total_walkers(self):
        raise NotImplementedError

    @property
    def total_steps(self):
        raise NotImplementedError

    @property
    def auto_correlations(self):
        raise NotImplementedError

    @classmethod
    def from_table(self, filename: str, model: ModelMapper, number_live_points: int = None):
        """
        Write a table of parameters, posteriors, priors and likelihoods

        Parameters
        ----------
        filename
            Where the table is to be written
        """

        sample_list = load_from_table(filename=filename)

        return OptimizerSamples(
            model=model,
            sample_list=sample_list
        )

    @property
    def info_json(self):
        return {
            "times": None,
            "check_size": self.auto_correlations.check_size,
            "required_length": self.auto_correlations.required_length,
            "change_threshold": self.auto_correlations.change_threshold,
            "total_walkers": self.total_walkers,
            "total_steps": self.total_steps,
            "time": self.time,
        }

    @property
    def pdf_converged(self):
        """
        To analyse and visualize samples using corner.py, the analysis must be sufficiently converged to produce
        smooth enough PDF for analysis. This property checks whether the non-linear search's samples are sufficiently
        converged for corner.py use.

        Emcee samples can be analysed by corner.py irrespective of how long the sampler has run, albeit low run times
        will likely produce inaccurate results.
        """
        try:
            samples_after_burn_in = self.samples_after_burn_in
            if len(samples_after_burn_in) == 0:
                return False
            return True
        except ValueError:
            return False

    @property
    def samples_after_burn_in(self) -> [list]:
        """
        The emcee samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters.
        """
        raise NotImplementedError()

    @property
    def converged(self) -> bool:
        """
        Whether the emcee samples have converged on a solution or if they are still in a burn-in period, based on the
        auto correlation times of parameters.
        """
        return self.auto_correlations.check_if_converged(
            total_samples=self.total_samples
        )

    @property
    def median_pdf_vector(self) -> [float]:
        """
        The median of the probability density function (PDF) of every parameter marginalized in 1D, returned
        as a list of values.

        This is computed by binning all sampls after burn-in into a histogram and take its median (e.g. 50%) value.
        """

        if self.pdf_converged:
            return [
                float(np.percentile(self.samples_after_burn_in[:, i], [50]))
                for i in range(self.model.prior_count)
            ]

        return self.max_log_likelihood_vector

    def vector_at_sigma(self, sigma: float) -> [float]:
        """
        The value of every parameter marginalized in 1D at an input sigma value of its probability density function
        (PDF), returned as two lists of values corresponding to the lower and upper values parameter values.

        For example, if sigma is 1.0, the marginalized values of every parameter at 31.7% and 68.2% percentiles of each
        PDF is returned.

        This does not account for covariance between parameters. For example, if two parameters (x, y) are degenerate
        whereby x decreases as y gets larger to give the same PDF, this function will still return both at their
        upper values. Thus, caution is advised when using the function to reperform a model-fits.

        For Emcee, if the samples have converged this is estimated by binning the samples after burn-in into a
        histogram and taking the parameter values at the input PDF %.

        Parameters
        ----------
        sigma
            The sigma within which the PDF is used to estimate errors (e.g. sigma = 1.0 uses 0.6826 of the PDF).
        """
        limit = math.erf(0.5 * sigma * math.sqrt(2))

        if self.pdf_converged:
            samples = self.samples_after_burn_in

            return [
                tuple(
                    np.percentile(samples[:, i], [100.0 * (1.0 - limit), 100.0 * limit])
                )
                for i in range(self.model.prior_count)
            ]

        parameters_min = list(
            np.min(self.parameter_lists[-self.unconverged_sample_size:], axis=0)
        )
        parameters_max = list(
            np.max(self.parameter_lists[-self.unconverged_sample_size:], axis=0)
        )

        return [
            (parameters_min[index], parameters_max[index])
            for index in range(len(parameters_min))
        ]

    @property
    def log_evidence(self):
        return None
