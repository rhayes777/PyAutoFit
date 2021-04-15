import numpy as np

from autoconf import conf
from autofit import exc
from autofit.non_linear import samples as samp
from autofit.non_linear.abstract_search import IntervalCounter
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.initializer import InitializerPrior


class AbstractNest(NonLinearSearch):
    def __init__(
            self,
            name=None,
            path_prefix=None,
            prior_passer=None,
            iterations_per_update=None,
            session=None,
            **kwargs
    ):
        """
        Abstract class of a nested sampling `NonLinearSearch` (e.g. MultiNest, Dynesty).

        **PyAutoFit** allows a nested sampler to automatically terminate when the acceptance ratio falls below an input
        threshold value. When this occurs, all samples are accepted using the current maximum log likelihood value,
        irrespective of how well the model actually fits the data.

        This feature should be used for non-linear searches where the nested sampler gets 'stuck', for example because
        the log likelihood function is stochastic or varies rapidly over small scales in parameter space. The results of
        samples using this feature are not realiable (given the log likelihood is being manipulated to end the run), but
        they are still valid results for linking priors to a new search and non-linear search.

        Parameters
        ----------
        prior_passer : af.PriorPasser
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        """
        super().__init__(
            name=name,
            path_prefix=path_prefix,
            prior_passer=prior_passer,
            initializer=InitializerPrior(),
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

    class Fitness(NonLinearSearch.Fitness):
        def __init__(
                self,
                paths,
                analysis,
                model,
                samples_from_model,
                stagger_resampling_likelihood,
                log_likelihood_cap=None,
                pool_ids=None
        ):

            super().__init__(
                paths=paths,
                analysis=analysis,
                model=model,
                samples_from_model=samples_from_model,
                log_likelihood_cap=log_likelihood_cap,
                pool_ids=pool_ids
            )

            self.stagger_resampling_likelihood = stagger_resampling_likelihood
            self.stagger_accepted_samples = 0
            self.resampling_figure_of_merit = -1.0e99

            self.should_check_terminate = IntervalCounter(1000)

        def __call__(self, parameters, *kwargs):

            try:
                return self.figure_of_merit_from_parameters(parameters=parameters)
            except exc.FitException:
                return self.stagger_resampling_figure_of_merit()

        def figure_of_merit_from_parameters(self, parameters):
            """The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. All Nested
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
            all initial samples returning the same log likelihood and the `NonLinearSearch` terminating."""

            if not self.stagger_resampling_likelihood:

                return self.resample_figure_of_merit

            else:

                if self.stagger_accepted_samples < 10:

                    self.stagger_accepted_samples += 1
                    self.resampling_figure_of_merit += 1e90

                    return self.resampling_figure_of_merit

                else:

                    return -1.0 * np.abs(self.resampling_figure_of_merit) * 10.0

    @property
    def config_type(self):
        return conf.instance["non_linear"]["nest"]

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None, pool_ids=None):

        return self.__class__.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_via_sampler_from_model,
            stagger_resampling_likelihood=self.config_dict_settings["stagger_resampling_likelihood"],
            log_likelihood_cap=log_likelihood_cap,
            pool_ids=pool_ids
        )

    def samples_via_csv_json_from_model(self, model):

        samples = self.paths.load_samples()
        samples_info = self.paths.load_samples_info()

        return samp.NestSamples(
            model=model,
            samples=samples,
            log_evidence=samples_info["log_evidence"],
            total_samples=samples_info["total_samples"],
            unconverged_sample_size=samples_info["unconverged_sample_size"],
            number_live_points=samples_info["number_live_points"],
            time=samples_info["time"],
        )
