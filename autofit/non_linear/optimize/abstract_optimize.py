from abc import ABC

from autoconf import conf
from autofit.non_linear import samples as samp
from autofit.non_linear.abstract_search import NonLinearSearch


class AbstractOptimizer(NonLinearSearch, ABC):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["optimize"]