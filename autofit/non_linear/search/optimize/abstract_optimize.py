from abc import ABC

from autoconf import conf
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.samples import Samples


class AbstractOptimizer(NonLinearSearch, ABC):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["optimize"]

    class Fitness(NonLinearSearch.Fitness):

        def figure_of_merit_from(self, parameter_list):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. `Emcee`
            uses the log posterior.

            L-BFGS uses the chi-squared value, which is the -2.0*log_posterior.
            """
            return -2.0 * self.log_posterior_from(parameter_list=parameter_list)

    @property
    def samples_cls(self):
        return Samples