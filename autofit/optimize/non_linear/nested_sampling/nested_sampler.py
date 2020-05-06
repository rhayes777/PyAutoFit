import logging
import os
import numpy as np

from autofit import conf
from autofit.optimize.non_linear import non_linear as nl
from autofit.optimize.non_linear.non_linear import Result
from autofit.optimize.non_linear.paths import Paths
from autofit.text import samples_text

from autofit import exc

logger = logging.getLogger(__name__)


class NestedSampler(nl.NonLinearOptimizer):
    def __init__(
        self,
        paths=None,
        sigma=3,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
    ):
        """
        Abstract class of a nested sampling non-linear search (e.g. MultiNest, Dynesty).

        **PyAutoFit** allows a nested sampler to automatically terminate when the acceptance ratio falls below an input
        threshold value. When this occurs, all samples are accepted using the current maximum log likelihood value,
        irrespective of how well the model actually fits the data.

        This feature should be used for non-linear searches where the nested sampler gets 'stuck', for example because
        the log likelihood function is stochastic or varies rapidly over small scales in parameter space. The results of
        samples using this feature are not realiable (given the log likelihood is being manipulated to end the run), but
        they are still valid results for linking priors to a new phase and non-linear search.

        Parameters
        ----------
        paths : af.Paths
            A class that manages all paths, e.g. where the phase outputs are stored, the non-linear search samples,
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

        sampler = conf.instance.non_linear.config_for("NestedSampler")

        self.terminate_at_acceptance_ratio = (
            sampler.get("settings", "terminate_at_acceptance_ratio", bool)
            if terminate_at_acceptance_ratio is None
            else terminate_at_acceptance_ratio
        )

        self.acceptance_ratio_threshold = (
            sampler.get("settings", "acceptance_ratio_threshold", float)
            if acceptance_ratio_threshold is None
            else acceptance_ratio_threshold
        )

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.sigma = self.sigma
        copy.terminate_at_acceptance_ratio = self.terminate_at_acceptance_ratio
        copy.acceptance_ratio_threshold = self.acceptance_ratio_threshold
        return copy

    class Fitness(nl.NonLinearOptimizer.Fitness):
        def __init__(
            self,
            paths,
            analysis,
            model,
            samples_from_model,
            terminate_at_acceptance_ratio,
            acceptance_ratio_threshold,
        ):

            super().__init__(
                paths=paths,
                analysis=analysis,
                model=model,
                samples_from_model=samples_from_model,
            )

            self.terminate_at_acceptance_ratio = terminate_at_acceptance_ratio
            self.acceptance_ratio_threshold = acceptance_ratio_threshold

            self.should_check_terminate = nl.IntervalCounter(1000)

        def __call__(self, params, *kwargs):

            self.check_terminate_sampling()

            try:

                instance = self.model.instance_from_vector(vector=params)
                return self.fit_instance(instance)

            except exc.FitException:

                return -np.inf

        def check_terminate_sampling(self):
            """Automatically terminate nested sampling when the sampler's acceptance ratio falls below a specified
            value. This termimation is performed by returning all log likelihoods as the currently value of the maximum
            log likelihood sample. This will lead to unreliable probability density functions and error estimates.

            The reason to use this function is for stochastic likelihood functions the sampler can determine the
            highest log likelihood models in parameter space but get 'stuck', unable to terminate as it cannot get all
            live points to within a small likelihood range of one another. Without this feature on the sampler will not
            end and suffer an extremely low acceptance rate.

            This check is performed every 1000 samples."""

            if self.terminate_at_acceptance_ratio:

                try:
                    samples = self.samples_from_model(model=self.model)
                except Exception:
                    samples = None

                try:

                    if (
                        samples.acceptance_ratio < self.acceptance_ratio_threshold
                    ) or self.terminate_has_begun:

                        self.terminate_has_begun = True

                        return self.max_log_likelihood

                except ValueError:

                    pass

    def _full_fit(self, model, analysis):

        if not os.path.exists(self.paths.has_completed_path):

            logger.info("Running Nested Sampler...")
            self._fit(model=model, analysis=analysis)
            logger.info("Nested Sampler complete")

            # TODO: Some of the results below use the backup_path, which isnt updated until the end if thiss function is
            # TODO: not located here. Do we need to rely just ono the optimizer foldeR? This is a good idea if we always
            # TODO: have a valid sym-link( e.g. even for aggregator use).

            self.paths.backup()
            open(self.paths.has_completed_path, "w+").close()
        else:
            logger.warning(f"{self.paths.phase_name} has run previously - skipping")

        samples = self.samples_from_model(model=model)

        instance = samples.max_log_likelihood_instance
        analysis.visualize(instance=instance, during_analysis=False)
        samples_text.results_to_file(
            samples=samples, file_results=self.paths.file_results, during_analysis=False
        )
        result = Result(samples=samples, previous_model=model)
        self.paths.backup_zip_remove()
        return result

    def fitness_function_from_model_and_analysis(self, model, analysis):

        return NestedSampler.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from_model,
            terminate_at_acceptance_ratio=self.terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=self.acceptance_ratio_threshold,
        )

    def samples_from_model(self, model):
        raise NotImplementedError()
