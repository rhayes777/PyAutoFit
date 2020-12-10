from abc import ABC

from autoconf import conf
from autofit.non_linear import samples as samp
from autofit.non_linear.abstract_search import NonLinearSearch


class AbstractOptimizer(NonLinearSearch, ABC):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["optimize"]

    def samples_via_csv_json_from_model(self, model):
        parameters, log_likelihoods, log_priors, log_posteriors, weights = samp.load_from_table(
            filename=self.paths.samples_file, model=model
        )

        return samp.OptimizerSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
        )
