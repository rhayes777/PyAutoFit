from abc import ABC

from autoconf import conf
from autofit import exc
from autofit.non_linear.abstract_search import NonLinearSearch



class AbstractOptimizer(NonLinearSearch, ABC):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["optimize"]

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

    def fitness_function_from_model_and_analysis(self, model, analysis, log_likelihood_cap=None):

        return self.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap
        )