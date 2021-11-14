from typing import Optional

import emcee
import numpy as np

from autofit import MCMCSamples
from autofit.mapper.model_mapper import ModelMapper
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings, AutoCorrelations
from autofit.non_linear.samples import Sample


class EmceeSamples(MCMCSamples):

    def __init__(
            self,
            model: ModelMapper,
            backend: emcee.backends.HDFBackend,
            auto_correlation_settings: AutoCorrelationsSettings,
            unconverged_sample_size: int = 100,
            time: Optional[float] = None,
    ):
        """
        Create a `Samples` object from this non-linear search's output files on the hard-disk and model.

        For Emcee, all quantities are extracted via the hdf5 backend of results.

        Attributes
        ----------
        total_walkers : int
            The total number of walkers used by this MCMC non-linear search.
        total_steps : int
            The total number of steps taken by each walker of this MCMC `NonLinearSearch` (the total samples is equal
            to the total steps * total walkers).
        """

        self.backend = backend

        parameter_lists = self.backend.get_chain(flat=True).tolist()

        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector)) for vector in parameter_lists
        ]

        log_posterior_list = self.backend.get_log_prob(flat=True).tolist()

        log_likelihood_list = [
            log_posterior - log_prior for
            log_posterior, log_prior in
            zip(log_posterior_list, log_prior_list)
        ]

        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            auto_correlation_settings=auto_correlation_settings,
            unconverged_sample_size=unconverged_sample_size,
            time=time,
        )

    @property
    def samples_after_burn_in(self) -> [list]:
        """
        The emcee samples with the initial burn-in samples removed.

        The burn-in period is estimated using the auto-correlation times of the parameters.
        """
        discard = int(3.0 * np.max(self.auto_correlations.times))
        thin = int(np.max(self.auto_correlations.times) / 2.0)
        return self.backend.get_chain(discard=discard, thin=thin, flat=True)

    @property
    def total_walkers(self):
        return len(self.backend.get_chain()[0, :, 0])

    @property
    def total_steps(self):
        return len(self.backend.get_log_prob())

    @property
    def auto_correlations(self):
        times = self.backend.get_autocorr_time(tol=0)

        previous_auto_correlation_times = emcee.autocorr.integrated_time(
            x=self.backend.get_chain()[: -self.auto_correlation_settings.check_size, :, :], tol=0
        )

        return AutoCorrelations(
            check_size=self.auto_correlation_settings.check_size,
            required_length=self.auto_correlation_settings.required_length,
            change_threshold=self.auto_correlation_settings.change_threshold,
            times=times,
            previous_times=previous_auto_correlation_times,
        )
