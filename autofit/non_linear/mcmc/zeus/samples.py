import logging
import numpy as np
from typing import List, Optional
import zeus

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples.mcmc import MCMCSamples
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings, AutoCorrelations
from autofit.non_linear.samples import Sample

logger = logging.getLogger(
    __name__
)

class ZeusSamples(MCMCSamples):

    @classmethod
    def from_results_internal(
            cls,
            results_internal,
            model: AbstractPriorModel,
            auto_correlation_settings: AutoCorrelationsSettings,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        The `Samples` classes in **PyAutoFit** provide an interface between the results of a `NonLinearSearch` (e.g.
        as files on your hard-disk) and Python.

        To create a `Samples` object after an `Zeus` model-fit the results must be converted from the
        native format used by `Zeus` (which is a HDFBackend) to lists of values, the format used by the **PyAutoFit**
        `Samples` objects.

        This classmethod performs this conversion before creating a `ZeusSamples` object.

        Parameters
        ----------
        results_internal
            The MCMC results in their native internal format from which the samples are computed.
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        auto_correlations_settings
            Customizes and performs auto correlation calculations performed during and after the search.
        unconverged_sample_size
            If the samples are for a search that is yet to convergence, a reduced set of samples are used to provide
            a rough estimate of the parameters. The number of samples is set by this parameter.
        time
            The time taken to perform the model-fit, which is passed around `Samples` objects for outputting
            information on the overall fit.
        """

        parameter_lists = results_internal.get_chain(flat=True).tolist()
        log_posterior_list = results_internal.get_log_prob(flat=True).tolist()
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)

        log_likelihood_list = [
            log_posterior - log_prior
            for log_posterior, log_prior
            in zip(log_posterior_list, log_prior_list)
        ]

        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        return ZeusSamples(
            model=model,
            sample_list=sample_list,
            auto_correlation_settings=auto_correlation_settings,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
            results_internal=results_internal,
        )

    @property
    def samples_after_burn_in(self) -> [List]:
        """
        The zeus samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters.
        """

        discard = int(3.0 * np.max(self.auto_correlations.times))
        thin = int(np.max(self.auto_correlations.times) / 2.0)
        return self.results_internal.get_chain(discard=discard, thin=thin, flat=True)

    @property
    def total_walkers(self):
        return len(self.results_internal.get_chain()[0, :, 0])

    @property
    def total_steps(self):
        return int(self.results_internal.ncall_total)

    @property
    def auto_correlations(self):

        times = zeus.AutoCorrTime(samples=self.results_internal.get_chain())
        try:
            previous_auto_correlation_times = zeus.AutoCorrTime(
                samples=self.results_internal.get_chain()[: -self.auto_correlation_settings.check_size, :, :],
            )
        except IndexError:
            logger.debug(
                "Unable to compute previous auto correlation times."
            )
            previous_auto_correlation_times = None

        return AutoCorrelations(
            check_size=self.auto_correlation_settings.check_size,
            required_length=self.auto_correlation_settings.required_length,
            change_threshold=self.auto_correlation_settings.change_threshold,
            times=times,
            previous_times=previous_auto_correlation_times,
        )
