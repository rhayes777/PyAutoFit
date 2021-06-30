from typing import Optional
from sqlalchemy.orm import Session

from autoconf import conf
from autofit import exc
from autofit.non_linear.abstract_search import NonLinearSearch


class AbstractMLE(NonLinearSearch):

    def __init__(
            self,
            name=None,
            path_prefix=None,
            unique_tag : Optional[str] = None,
            prior_passer=None,
            initializer=None,
            iterations_per_update : int = None,
            session : Optional[Session] = None,
            **kwargs
    ):

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            prior_passer=prior_passer,
            initializer=initializer,
            iterations_per_update=iterations_per_update,
            session=session,
            **kwargs
        )

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mle"]

    class Fitness(NonLinearSearch.Fitness):
        def __call__(self, parameters):
            try:
                return self.figure_of_merit_from(parameter_list=parameters)
            except exc.FitException:
                return self.resample_figure_of_merit

        def figure_of_merit_from(self, parameter_list):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. `Emcee`
            uses the log posterior.

            L-BFGS uses the chi-squared value, which is the -2.0*log_posterior.
            """
            return -2.0 * self.log_posterior_from(parameter_list=parameter_list)