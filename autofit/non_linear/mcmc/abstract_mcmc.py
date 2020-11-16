import json

from autoconf import conf
from autofit.non_linear import samples as samp
from autofit.non_linear.abstract_search import NonLinearSearch


class AbstractMCMC(NonLinearSearch):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["mcmc"]

    def samples_via_csv_json_from_model(self, model):
        parameters, log_likelihoods, log_priors, log_posteriors, weights = samp.load_from_table(
            filename=self.paths.samples_file, model=model
        )

        with open(self.paths.info_file) as infile:
            samples_info = json.load(infile)

        return samp.MCMCSamples(
            model=model,
            parameters=parameters,
            log_likelihoods=log_likelihoods,
            log_priors=log_priors,
            weights=weights,
            auto_correlation_times=samples_info["auto_correlation_times"],
            auto_correlation_check_size=samples_info["auto_correlation_check_size"],
            auto_correlation_required_length=samples_info[
                "auto_correlation_required_length"
            ],
            auto_correlation_change_threshold=samples_info[
                "auto_correlation_change_threshold"
            ],
            total_walkers=samples_info["total_walkers"],
            total_steps=samples_info["total_steps"],
            time=samples_info["time"],
        )
