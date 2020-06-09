import os

import pytest

import autofit as af
from autofit import Paths

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture(name="downhill_simplex")
def make_downhill_simplex():
    def fmin(fitness_function, x0):
        fitness_function(x0)
        return x0

    return af.DownhillSimplex(
        fmin=fmin,
        paths=Paths(name="name", folders=("folders",), tag="tag"),
    )


# class TestDownhillSimplex:
#     def test_instance(self, downhill_simplex, model):
#         model.mock_class = MockClassNLOx4()
#
#         assert hasattr(model.instance_from_unit_vector([]), "mock_class")
#
#         result = downhill_simplex.fit(MockAnalysis(), model)
#
#         assert result.instance.mock_class.one == 1
#         assert result.instance.mock_class.two == 2
#         assert result.log_likelihood == 1
#
#     def test_model(self, downhill_simplex, model):
#         model.mock_class = af.PriorModel(MockClassNLOx4)
#         result = downhill_simplex.fit(MockAnalysis(), model)
#
#         assert result.instance.mock_class.one == 0.0
#         assert result.instance.mock_class.two == 0.0
#         assert result.log_likelihood == 1
#
#         assert result.model.mock_class.one.mean == 0.0
#         assert result.model.mock_class.two.mean == 0.0
#
#     def test_instance_and_model(self, downhill_simplex, model):
#         model.instance = MockClassNLOx4()
#         model.model = af.PriorModel(MockClassNLOx4)
#
#         result = downhill_simplex.fit(MockAnalysis(), model)
#
#         assert result.instance.instance.one == 1
#         assert result.instance.instance.two == 2
#         assert result.instance.model.one == 0.0
#         assert result.instance.model.two == 0.0
#         assert result.model.model.one.mean == 0.0
#         assert result.model.model.two.mean == 0.0
#         assert result.log_likelihood == 1


class TestCopyWithNameExtension:
    @staticmethod
    def assert_non_linear_attributes_equal(copy, search):
        assert copy.paths.name == "phase_name/one"

    def test_downhill_simplex(self):
        search = af.DownhillSimplex(Paths("phase_name"), fmin=lambda x: x)

        copy = search.copy_with_name_extension("one")
        self.assert_non_linear_attributes_equal(copy, search)
        assert isinstance(copy, af.DownhillSimplex)
        assert copy.fmin is search.fmin
        assert copy.xtol is search.xtol
        assert copy.ftol is search.ftol
        assert copy.maxiter is search.maxiter
        assert copy.maxfun is search.maxfun
        assert copy.full_output is search.full_output
        assert copy.disp is search.disp
        assert copy.retall is search.retall
