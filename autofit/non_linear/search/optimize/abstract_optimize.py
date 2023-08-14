from abc import ABC

from autoconf import conf
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.samples import Samples


class AbstractOptimizer(NonLinearSearch, ABC):
    @property
    def config_type(self):
        return conf.instance["non_linear"]["optimize"]

    @property
    def samples_cls(self):
        return Samples