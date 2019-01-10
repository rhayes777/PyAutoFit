from autofit.core import prior as p
from autofit.core import non_linear
from autofit.core import model_mapper


class TestGridSearch(object):
    def test_1d(self):
        grid_search = non_linear.GridSearch(step_size=0.1)
        grid_search.variable.one = p.UniformPrior()
