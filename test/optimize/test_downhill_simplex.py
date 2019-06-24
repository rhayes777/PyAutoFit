import os

import pytest


import autofit as af
import autofit as af
import autofit as af
from test.mock import MockClassNLOx4, MockAnalysis

pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/configs/non_linear".format(
            os.path.dirname(os.path.realpath(__file__))))


@pytest.fixture(name="downhill_simplex")
def make_downhill_simplex():
    def fmin(fitness_function, x0):
        fitness_function(x0)
        return x0

    return af.DownhillSimplex(
        fmin=fmin,
        phase_name='name',
        phase_folders='folders',
        phase_tag='tag',
        model_mapper=af.ModelMapper()
    )


class TestDownhillSimplex(object):

    def test_constant(self, downhill_simplex):
        downhill_simplex.variable.mock_class = MockClassNLOx4()

        print(downhill_simplex.variable.instance_tuples)

        assert len(downhill_simplex.variable.instance_tuples) == 1
        assert hasattr(downhill_simplex.variable.instance_from_unit_vector([]),
                       "mock_class")

        result = downhill_simplex.fit(MockAnalysis())

        assert result.constant.mock_class.one == 1
        assert result.constant.mock_class.two == 2
        assert result.figure_of_merit == 1

    def test_variable(self, downhill_simplex):
        downhill_simplex.variable.mock_class = af.PriorModel(
            MockClassNLOx4)
        result = downhill_simplex.fit(MockAnalysis())

        assert result.constant.mock_class.one == 0.0
        assert result.constant.mock_class.two == 0.0
        assert result.figure_of_merit == 1

        assert result.variable.mock_class.one.mean == 0.0
        assert result.variable.mock_class.two.mean == 0.0

    def test_constant_and_variable(self, downhill_simplex):
        downhill_simplex.variable.constant = MockClassNLOx4()
        downhill_simplex.variable.variable = af.PriorModel(
            MockClassNLOx4)

        result = downhill_simplex.fit(MockAnalysis())

        assert result.constant.constant.one == 1
        assert result.constant.constant.two == 2
        assert result.constant.variable.one == 0.0
        assert result.constant.variable.two == 0.0
        assert result.variable.variable.one.mean == 0.0
        assert result.variable.variable.two.mean == 0.0
        assert result.figure_of_merit == 1


class TestCopyWithNameExtension(object):

    @staticmethod
    def assert_non_linear_attributes_equal(copy, optimizer):
        assert copy.phase_name == "phase_name/one"
        assert copy.variable == optimizer.variable

    def test_downhill_simplex(self):
        optimizer = af.DownhillSimplex("phase_name", fmin=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy,
                          af.DownhillSimplex)
        assert copy.fmin is optimizer.fmin
        assert copy.xtol is optimizer.xtol
        assert copy.ftol is optimizer.ftol
        assert copy.maxiter is optimizer.maxiter
        assert copy.maxfun is optimizer.maxfun
        assert copy.full_output is optimizer.full_output
        assert copy.disp is optimizer.disp
        assert copy.retall is optimizer.retall
