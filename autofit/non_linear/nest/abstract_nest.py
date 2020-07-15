import numpy as np

from autoconf import conf
from autofit.non_linear.initializer import InitializerPrior
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.abstract_search import IntervalCounter
from autofit.non_linear.paths import Paths

from autofit import exc

class AbstractNest(NonLinearSearch):
    def __init__(
        self,
        paths=None,
        prior_passer=None,
        iterations_per_update=None,
        terminate_at_acceptance_ratio=None,
        acceptance_ratio_threshold=None,
        stagger_resampling_likelihood=None,
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
            Manages all paths, e.g. where the search outputs are stored, the samples, backups, etc.
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this non-linear search to a subsequent non-linear search.
        terminate_at_acceptance_ratio : bool
            If *True*, the sampler will automatically terminate when the acceptance ratio falls behind an input
            threshold value.
        acceptance_ratio_threshold : float
            The acceptance ratio threshold below which sampling terminates if *terminate_at_acceptance_ratio* is *True*.
        """

        if paths is None:
            paths = Paths()

        super().__init__(
            paths=paths,
            prior_passer=prior_passer,
            initializer=InitializerPrior(),
            iterations_per_update=iterations_per_update,
        )

        self.terminate_at_acceptance_ratio = (
            self._config("settings", "terminate_at_acceptance_ratio", bool)
            if terminate_at_acceptance_ratio is None
            else terminate_at_acceptance_ratio
        )

        self.acceptance_ratio_threshold = (
            self._config("settings", "acceptance_ratio_threshold", float)
            if acceptance_ratio_threshold is None
            else acceptance_ratio_threshold
        )

        self.stagger_resampling_likelihood = (
            self._config("settings", "stagger_resampling_likelihood", bool)
            if stagger_resampling_likelihood is None
            else stagger_resampling_likelihood
        )

    class Fitness(NonLinearSearch.Fitness):
        def __init__(
            self,
            paths,
            analysis,
            model,
            samples_from_model,
            stagger_resampling_likelihood,
            terminate_at_acceptance_ratio,
            acceptance_ratio_threshold,
        ):

            super().__init__(
                paths=paths,
                analysis=analysis,
                model=model,
                samples_from_model=samples_from_model,
            )

            self.stagger_resampling_likelihood = stagger_resampling_likelihood
            self.stagger_accepted_samples = 0
            self.resampling_figure_of_merit = -1.0e99

            self.terminate_at_acceptance_ratio = terminate_at_acceptance_ratio
            self.acceptance_ratio_threshold = acceptance_ratio_threshold

            self.should_check_terminate = IntervalCounter(1000)

        def __call__(self, parameters, *kwargs):

            self.check_terminate_sampling()

            try:
                return self.figure_of_merit_from_parameters(parameters=parameters)
            except exc.FitException:
                return self.stagger_resampling_figure_of_merit()

        def figure_of_merit_from_parameters(self, parameters):
            """The figure of merit is the value that the non-linear search uses to sample parameter space. All Nested
            samplers use the log likelihood.
            """
            try:
                return self.log_likelihood_from_parameters(parameters=parameters)
            except exc.FitException:
                raise exc.FitException

        def stagger_resampling_figure_of_merit(self):
            """By default, when a fit raises an exception a log likelihood of -np.inf is returned, which leads the
            sampler to discard the sample.

            However, we found that this causes memory issues when running PyMultiNest. Therefore, we 'hack' a solution
            by not returning -np.inf (which leads the sample to be discarded) but instead a large negative float which
            is treated as a real sample (and does not lead too memory issues). The value returned is staggered to avoid
            all initial samples returning the same log likelihood and the non-linear search terminating."""

            if not self.stagger_resampling_likelihood:

                return self.resample_figure_of_merit

            else:

                if self.stagger_accepted_samples < 10:

                    self.stagger_accepted_samples += 1
                    self.resampling_figure_of_merit += 1e90

                    return self.resampling_figure_of_merit

                else:

                    return -1.0 * np.abs(self.resampling_figure_of_merit) * 10.0

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

    @property
    def config_type(self):
        return conf.instance.nest

    def copy_with_name_extension(self, extension, remove_phase_tag=False):
        copy = super().copy_with_name_extension(
            extension=extension, remove_phase_tag=remove_phase_tag
        )
        copy.prior_passer = self.prior_passer
        copy.terminate_at_acceptance_ratio = self.terminate_at_acceptance_ratio
        copy.acceptance_ratio_threshold = self.acceptance_ratio_threshold
        copy.stagger_resampling_likelihood = self.stagger_resampling_likelihood
        return copy

    def fitness_function_from_model_and_analysis(self, model, analysis):

        return self.__class__.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from_model,
            stagger_resampling_likelihood=self.stagger_resampling_likelihood,
            terminate_at_acceptance_ratio=self.terminate_at_acceptance_ratio,
            acceptance_ratio_threshold=self.acceptance_ratio_threshold,
        )

    def samples_from_model(self, model):
        raise NotImplementedError()
