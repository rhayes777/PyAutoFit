from autoconf import conf

from autofit.non_linear.abstract_search import NonLinearSearch


class AbstractOptimizer(NonLinearSearch):
    @property
    def config_type(self):
        return conf.instance.optimize
