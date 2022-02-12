from typing import Optional

import numpy as np

from autoconf import conf
from autofit import exc
from autofit.database.sqlalchemy_ import sa
from autofit.non_linear.abstract_search import IntervalCounter
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.abstract_search import PriorPasser
from autofit.non_linear.initializer import InitializerPrior


class AbstractNest(NonLinearSearch):
    def __init__(
            self,
            name: Optional[str] = None,
            path_prefix: Optional[str] = None,
            unique_tag: Optional[str] = None,
            prior_passer: Optional[PriorPasser] = None,
            iterations_per_update: Optional[int] = None,
            session: Optional[sa.orm.Session] = None,
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
        prior_passer
            Controls how priors are passed from the results of this `NonLinearSearch` to a subsequent non-linear search.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """
        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
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
                log_likelihood_cap=None
        ):

            super().__init__(
                paths=paths,
                analysis=analysis,
                model=model,
                samples_from_model=samples_from_model,
                log_likelihood_cap=log_likelihood_cap
            )

            self.stagger_accepted_samples = 0
            self.resampling_figure_of_merit = -1.0e99

            self.should_check_terminate = IntervalCounter(1000)

        def __call__(self, parameters, *kwargs):

            try:
                return self.figure_of_merit_from(parameter_list=parameters)
            except exc.FitException:
                return self.resample_figure_of_merit

        def figure_of_merit_from(self, parameter_list):
            """The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. All Nested
            samplers use the log likelihood.
            """
            return self.log_likelihood_from(parameter_list=parameter_list)

    @property
    def config_type(self):
        return conf.instance["non_linear"]["nest"]

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None):

        return self.__class__.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap
        )
