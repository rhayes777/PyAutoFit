import json

from autoconf import conf

from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear import samples as samp

class AbstractOptimizer(NonLinearSearch):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["optimize"]

    def samples_via_csv_json_from_model(self, model):

        parameters, log_likelihoods, log_priors, log_posteriors, weights = samp.load_from_table(
            filename=self.paths.samples_file, model=model
        )

        with open(self.paths.info_file) as infile:
            samples_info = json.load(infile)

        return samp.OptimizerSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
        )