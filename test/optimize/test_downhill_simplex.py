import os

import pytest

import autofit.mapper.prior_model
import autofit.optimize.non_linear.downhill_simplex
import autofit.optimize.non_linear.grid_search
import autofit.optimize.non_linear.multi_nest
import autofit.optimize.non_linear.non_linear
from autofit import conf
from autofit.mapper import model_mapper

from test.mock.mock import MockClassNLOx4, MockAnalysis

pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')

@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config(
        "{}/../test_files/configs/non_linear".format(os.path.dirname(os.path.realpath(__file__))))

@pytest.fixture(name="downhill_simplex")
def make_downhill_simplex():
    def fmin(fitness_function, x0):
        fitness_function(x0)
        return x0

    return autofit.optimize.non_linear.downhill_simplex.DownhillSimplex(fmin=fmin, phase_name='', model_mapper=model_mapper.ModelMapper())

class TestDownhillSimplex(object):

    def test_constant(self, downhill_simplex):
        downhill_simplex.variable.mock_class = MockClassNLOx4()

        assert len(downhill_simplex.variable.instance_tuples) == 1
        assert hasattr(downhill_simplex.variable.instance_from_unit_vector([]), "mock_class")

        result = downhill_simplex.fit(MockAnalysis())

        assert result.constant.mock_class.one == 1
        assert result.constant.mock_class.two == 2
        assert result.figure_of_merit == 1

    def test_variable(self, downhill_simplex):
        downhill_simplex.variable.mock_class = autofit.mapper.prior_model.PriorModel(MockClassNLOx4)
        result = downhill_simplex.fit(MockAnalysis())

        assert result.constant.mock_class.one == 0.0
        assert result.constant.mock_class.two == 0.0
        assert result.figure_of_merit == 1

        assert result.variable.mock_class.one.mean == 0.0
        assert result.variable.mock_class.two.mean == 0.0

    def test_constant_and_variable(self, downhill_simplex):
        downhill_simplex.variable.constant = MockClassNLOx4()
        downhill_simplex.variable.variable = autofit.mapper.prior_model.PriorModel(MockClassNLOx4)

        result = downhill_simplex.fit(MockAnalysis())

        assert result.constant.constant.one == 1
        assert result.constant.constant.two == 2
        assert result.constant.variable.one == 0.0
        assert result.constant.variable.two == 0.0
        assert result.variable.variable.one.mean == 0.0
        assert result.variable.variable.two.mean == 0.0
        assert result.figure_of_merit == 1