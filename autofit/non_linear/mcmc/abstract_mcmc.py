import json

from autoconf import conf
from autofit.non_linear import samples as samp
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings, AutoCorrelations


class AbstractMCMC(NonLinearSearch):

    def __init__(
            self,
            name="",
            path_prefix="",
            prior_passer=None,
            initializer=None,
            auto_correlations_settings = AutoCorrelationsSettings(),
            iterations_per_update=None,
            number_of_cores=None,
            session=None,
            **kwargs
    ):

        self.auto_correlations_settings = auto_correlations_settings
        self.auto_correlations_settings.update_via_config(
            config=self.config_type[self.__class__.__name__]["auto_correlations"]
        )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            session=session,
            number_of_cores=number_of_cores,
            **kwargs
        )

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mcmc"]

    def samples_via_csv_json_from_model(self, model):

        samples = self.paths.load_samples()
        samples_info = self.paths.load_samples_info()

        return samp.MCMCSamples(
            model=model,
            samples=samples,
            auto_correlations=AutoCorrelations(
                check_size=samples_info["check_size"],
                required_length=samples_info["required_length"],
                change_threshold=samples_info["change_threshold"],
                times=None,
                previous_times=None
            ),
            total_walkers=samples_info["total_walkers"],
            total_steps=samples_info["total_steps"],
            time=samples_info["time"],
        )
