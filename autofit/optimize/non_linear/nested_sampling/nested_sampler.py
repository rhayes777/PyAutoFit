import logging
import os

from autofit import conf
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.samples import AbstractSamples
from autofit.optimize.non_linear.paths import Paths

logger = logging.getLogger(__name__)


class NestedSampler(NonLinearOptimizer):
    def __init__(self, paths=None, sigma=3):
        """
        Abstract class of a nested sampling non-linear search (e.g. MultiNest, Dynesty).

        **PyAutoFit** allows a nested sampler to automatically terminate when the acceptance ratio falls below an input
        threshold value. When this occurs, all samples are accepted using the current maximum log likelihood value,
        irrespective of how well the model actually fits the data.

        This feature should be used for non-linear searches where the nested sampler gets 'stuck', for example because
        the log likelihood function is stochastic or varies rapidly over small scales in parameter space. The results of
        chains using this feature are not realiable (given the log likelihood is being manipulated to end the run), but
        they are still valid results for linking priors to a new phase and non-linear search.

        Parameters
        ----------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search chains,
            backups, etc.
        sigma : float
            The error-bound value that linked Gaussian prior withs are computed using. For example, if sigma=3.0,
            parameters will use Gaussian Priors with widths coresponding to errors estimated at 3 sigma confidence.
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value.
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is *True*.
        """

        if paths is None:
            paths = Paths(non_linear_name=type(self).__name__.lower())

        super().__init__(paths)

        self.sigma = sigma

        self.terminate_at_acceptance_ratio = self.config(
            "terminate_at_acceptance_ratio", bool
        )
        self.acceptance_ratio_threshold = self.config(
            "acceptance_ratio_threshold", float
        )

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
        copy.terminate_at_acceptance_ratio = self.terminate_at_acceptance_ratio
        copy.acceptance_ratio_threshold = self.acceptance_ratio_threshold
        return copy

    class Fitness(NonLinearOptimizer.Fitness):
        def __init__(
            self,
            paths,
            analysis,
            model,
            samples_from_model,
            terminate_at_acceptance_ratio,
            acceptance_ratio_threshold,
        ):
            super().__init__(paths=paths, analysis=analysis, model=model, samples_from_model=samples_from_model)
            self.accepted_samples = 0

            self.model_results_output_interval = conf.instance.general.get(
                "output", "model_results_output_interval", int
            )
            self.terminate_at_acceptance_ratio = terminate_at_acceptance_ratio
            self.acceptance_ratio_threshold = acceptance_ratio_threshold

            self.terminate_has_begun = False

        def __call__(self, instance):
            if self.terminate_at_acceptance_ratio:
                if os.path.isfile(self.paths.file_summary):
                    try:
                        if (
                            self.samples.acceptance_ratio
                            < self.acceptance_ratio_threshold
                        ) or self.terminate_has_begun:
                            self.terminate_has_begun = True
                            return self.max_likelihood
                    except ValueError:
                        pass

            return self.fit_instance(instance)

    def _fit(self, analysis, model):

        if not os.path.exists(self.paths.has_completed_path):

            fitness_function = NestedSampler.Fitness(
                paths=self.paths,
                analysis=analysis,
                model=model,
                samples_from_model=self.samples_from_model,
                terminate_at_acceptance_ratio=self.terminate_at_acceptance_ratio,
                acceptance_ratio_threshold=self.acceptance_ratio_threshold,
            )

            logger.info("Running Nested Sampler...")
            self._simple_fit(model, fitness_function.__call__)
            logger.info("Nested Sampler complete")

            # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
            # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
            # TODO: have a valid sym-link( e.g. even for aggregator use).

            self.paths.backup()
            open(self.paths.has_completed_path, "w+").close()
        else:
            logger.warning(f"{self.paths.phase_name} has run previously - skipping")

        samples = self.samples_from_model(model=model, paths=self.paths)

        instance = samples.max_log_likelihood_instance
        analysis.visualize(instance=instance, during_analysis=False)
        samples.output_results(during_analysis=False)
        samples.output_pdf_plots()
        result = Result(
            instance=instance,
            log_likelihood=samples.max_log_posterior,
            output=samples,
            previous_model=model,
            gaussian_tuples=samples.gaussian_priors_at_sigma(self.sigma),
        )
        self.paths.backup_zip_remove()
        return result

    def samples_from_model(self, model, paths):
        return NotImplementedError()


