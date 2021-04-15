import json

from autoconf import conf
from autofit.non_linear import samples as samp
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.mcmc.auto_correlations import AutoCorrelationsSettings, AutoCorrelations


class AbstractMCMC(NonLinearSearch):

    def __init__(
            self,
            name=None,
            path_prefix=None,
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
            number_of_cores=number_of_cores,
            session=session,
            **kwargs
        )

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mcmc"]
