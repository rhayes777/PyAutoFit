import pytest

import autofit as af
import autofit.graphical as g


class Analysis(af.Analysis):
    def __init__(self, value):
        self.value = value

    def log_likelihood_function(self, instance):
        return -(instance.one - self.value) ** 2


@pytest.fixture(name="model_factor")
def make_model_factor():
    model = af.Collection(one=af.UniformPrior())

    return g.ModelFactor(model, Analysis(0.5))


@pytest.fixture(name="model_factor_2")
def make_model_factor_2():
    model_2 = af.Collection(one=af.UniformPrior())

    return g.ModelFactor(model_2, Analysis(0.0))


class TestGlobalLikelihood:
    @pytest.mark.parametrize("unit_value, likelihood", [(0.5, 0.0), (0.0, -0.25)])
    def test_single_factor(self, model_factor, unit_value, likelihood):
        assert (
                model_factor.log_likelihood_function(
                    model_factor.global_prior_model.instance_from_unit_vector([unit_value])
                )
                == likelihood
        )

    @pytest.mark.parametrize("unit_value, likelihood", [(0.5, 0.0), (0.0, -0.0625)])
    def test_collection(self, model_factor, unit_value, likelihood):
        collection = g.GraphicalModel(model_factor, model_factor)
        assert (
                collection.log_likelihood_function(
                    collection.global_prior_model.instance_from_unit_vector([unit_value])
                )
                == likelihood
        )

    @pytest.mark.parametrize(
        "unit_vector, likelihood", [([0.5, 0.0], 0.0), ([1.0, 0.5], -0.0625)]
    )
    def test_two_factor(self, model_factor, model_factor_2, unit_vector, likelihood):
        collection = g.GraphicalModel(model_factor, model_factor_2)

        assert (
                collection.log_likelihood_function(
                    collection.global_prior_model.instance_from_unit_vector(unit_vector)
                )
                == likelihood
        )

    def test_global_search(self, model_factor, model_factor_2):
        collection = g.GraphicalModel(model_factor, model_factor_2)
        search = af.MockSearch()

        class Analysis(af.Analysis):
            def log_likelihood_function(self, instance):
                return collection.log_likelihood_function(instance)

        search.fit(collection.global_prior_model, Analysis())
