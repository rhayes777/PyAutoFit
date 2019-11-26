import os

import pytest

import autofit as af
from autofit import Paths
from test_autofit.mock import MockClassNLOx4, MockAnalysis

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/configs/non_linear".format(
            os.path.dirname(os.path.realpath(__file__))
        )
    )


@pytest.fixture(name="downhill_simplex")
def make_downhill_simplex():
    def fmin(fitness_function, x0):
        fitness_function(x0)
        return x0

    return af.DownhillSimplex(
        fmin=fmin,
        paths=Paths(phase_name="name", phase_folders=("folders",), phase_tag="tag"),
    )


class TestDownhillSimplex(object):
    def test_instance(self, downhill_simplex, model):
        model.mock_class = MockClassNLOx4()

        assert hasattr(model.instance_from_unit_vector([]), "mock_class")

        result = downhill_simplex.fit(MockAnalysis(), model)

        assert result.instance.mock_class.one == 1
        assert result.instance.mock_class.two == 2
        assert result.figure_of_merit == 1

    def test_model(self, downhill_simplex, model):
        model.mock_class = af.PriorModel(MockClassNLOx4)
        result = downhill_simplex.fit(MockAnalysis(), model)

        assert result.instance.mock_class.one == 0.0
        assert result.instance.mock_class.two == 0.0
        assert result.figure_of_merit == 1

        assert result.model.mock_class.one.mean == 0.0
        assert result.model.mock_class.two.mean == 0.0

    def test_instance_and_model(self, downhill_simplex, model):
        model.instance = MockClassNLOx4()
        model.model = af.PriorModel(MockClassNLOx4)

        result = downhill_simplex.fit(MockAnalysis(), model)

        assert result.instance.instance.one == 1
        assert result.instance.instance.two == 2
        assert result.instance.model.one == 0.0
        assert result.instance.model.two == 0.0
        assert result.model.model.one.mean == 0.0
        assert result.model.model.two.mean == 0.0
        assert result.figure_of_merit == 1


class TestCopyWithNameExtension(object):
    @staticmethod
    def assert_non_linear_attributes_equal(copy, optimizer):
        assert copy.paths.phase_name == "phase_name/one"

    def test_downhill_simplex(self):
        optimizer = af.DownhillSimplex(Paths("phase_name"), fmin=lambda x: x)

        copy = optimizer.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, optimizer)
        assert isinstance(copy, af.DownhillSimplex)
        assert copy.fmin is optimizer.fmin
        assert copy.xtol is optimizer.xtol
        assert copy.ftol is optimizer.ftol
        assert copy.maxiter is optimizer.maxiter
        assert copy.maxfun is optimizer.maxfun
        assert copy.full_output is optimizer.full_output
        assert copy.disp is optimizer.disp
        assert copy.retall is optimizer.retall
