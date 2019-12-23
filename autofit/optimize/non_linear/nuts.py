import logging

import numpy as np
from autofit.optimize.non_linear.nuts_src import emcee_nuts

from autofit import conf, exc
from autofit.optimize.non_linear.output import AbstractOutput
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.non_linear import persistent_timer

logger = logging.getLogger(__name__)


class NUTS(NonLinearOptimizer):
    def __init__(self, paths, sigma_limit=3):
        """
        Class to setup and run a MultiNest lens and output the MultiNest nlo.

        This interfaces with an input model_mapper, which is used for setting up the \
        individual model instances that are passed to each iteration of MultiNest.
        """

        super().__init__(paths)

        self.sigma_limit = sigma_limit

        logger.debug("Creating NUTS NLO")

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma_limit = self.sigma_limit
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(
            self, paths, analysis, instance_from_physical_vector, output_results
        ):
            super().__init__(paths, analysis, output_results)
            self.instance_from_physical_vector = instance_from_physical_vector
            self.accepted_samples = 0

            self.model_results_output_interval = conf.instance.general.get(
                "output", "model_results_output_interval", int
            )

        def fit_instance(self, instance):
            likelihood = self.analysis.fit(instance)

            # if likelihood > self.max_likelihood:
            #
            #     self.max_likelihood = likelihood
            #     self.result = Result(instance, likelihood)
            #
            #     if self.should_visualize():
            #         self.analysis.visualize(instance, during_analysis=True)
            #
            #     if self.should_backup():
            #         self.paths.backup()
            #
            #     if self.should_output_model_results():
            #         self.output_results(during_analysis=True)

            return likelihood

        def __call__(self, cube):

            try:
                instance = self.instance_from_physical_vector(cube)
                likelihood = self.fit_instance(instance)

                print()
                print(cube)
                print()
                print(likelihood)

            except exc.FitException:

                likelihood = -1.0e8

            return likelihood

    @persistent_timer
    def fit(self, analysis, model):
        output = AbstractOutput(model=model, paths=self.paths)

        output.save_model_info()

        fitness_function = NUTS.Fitness(
            paths=self.paths,
            analysis=analysis,
            instance_from_physical_vector=model.instance_from_physical_vector,
            output_results=None,
        )

        nuts_sampler = emcee_nuts.NUTSSampler(
            dim=model.prior_count,
            log_prob_fn=fitness_function.__call__,
            gradient_fn=None,
        )

        initial_state = model.physical_values_from_prior_medians

        logger.info("Running NUTS Sampling...")
        samples = nuts_sampler.run_mcmc(
            initial_state=initial_state, steps=5000, steps_burn_in=5000, delta=0.6
        )
        logger.info("NUTS complete")

        print(samples)
        stop

        # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
        # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
        # TODO: have a valid sym-link( e.g. even for aggregator use).

        self.paths.backup()
        #    instance = output.most_likely_model_instance
        #    analysis.visualize(instance=instance, during_analysis=False)
        #    output.output_results(during_analysis=False)
        #    output.output_pdf_plots()
        #     result = Result(
        #         instance=instance,
        #         figure_of_merit=output.evidence,
        #         previous_model=model,
        #         gaussian_tuples=output.gaussian_priors_at_sigma_limit(
        #             self.sigma_limit
        #         ),
        #     )
        #     self.paths.backup_zip_remove()
        return None
