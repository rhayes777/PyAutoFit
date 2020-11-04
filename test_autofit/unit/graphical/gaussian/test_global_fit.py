import pytest

import autofit as af
import autofit.graphical as g


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Collection(
        one=af.UniformPrior()
    )


@pytest.fixture(
    name="model_factor"
)
def make_model_factor(model):
    def likelihood_function(
            instance
    ):
        return -(instance.one - 0.5) ** 2

    return g.ModelFactor(
        model,
        likelihood_function
    )


class TestGlobalLikelihood:
    @pytest.mark.parametrize(
        "unit_value, likelihood",
        [
            (0.5, 0.0),
            (0.0, -0.25)
        ]
    )
    def test_single_factor(
            self,
            model_factor,
            model,
            unit_value,
            likelihood
    ):
        assert model_factor.global_likelihood(
            model.instance_from_unit_vector(
                [unit_value]
            )
        ) == likelihood

    @pytest.mark.parametrize(
        "unit_value, likelihood",
        [
            (0.5, 0.0),
            (0.0, -0.0625)
        ]
    )
    def test_collection(
            self,
            model_factor,
            model,
            unit_value,
            likelihood
    ):
        collection = g.ModelFactorCollection(
            model_factor,
            model_factor
        )
        assert collection.global_likelihood(
            model.instance_from_unit_vector(
                [unit_value]
            )
        ) == likelihood
