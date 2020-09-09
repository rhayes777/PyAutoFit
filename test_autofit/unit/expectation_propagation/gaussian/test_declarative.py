import numpy as np
import pytest

import autofit as af
import autofit.expectation_propagation as ep
from test_autofit.unit.expectation_propagation.gaussian.model import Gaussian, make_data, _likelihood


def test_shared_intensity():
    n_observations = 100
    x = np.arange(n_observations)

    intensity = 25.0
    intensity_prior = af.GaussianPrior(
        mean=25,
        sigma=10
    )

    def make_factor_model(
            centre,
            sigma
    ):
        y = make_data(
            Gaussian(
                centre=centre,
                intensity=intensity,
                sigma=sigma
            ),
            x
        )
        prior_model = af.PriorModel(
            Gaussian,
            centre=af.GaussianPrior(
                mean=50,
                sigma=20
            ),
            intensity=intensity_prior,
            sigma=af.GaussianPrior(
                mean=10,
                sigma=10
            )
        )

        def likelihood_function(instance):
            y_model = instance(x)
            return np.mean(
                _likelihood(
                    y_model,
                    y
                )
            )

        return ep.LikelihoodModel(
            prior_model,
            likelihood_function=likelihood_function
        )

    factor_model = make_factor_model(
        centre=40,
        sigma=10
    ) * make_factor_model(
        centre=60,
        sigma=15
    )

    assert len(factor_model.message_dict) == 5
    assert len(factor_model.graph.factors) == 7

    mean_field_approximation = factor_model.mean_field_approximation

    opt = ep.optimise.LaplaceOptimiser(
        mean_field_approximation,
        n_iter=3
    )

    opt.run()

    for variable in factor_model.prior_variables:
        print(f"{variable.name} = {opt.model_approx[variable].mu}")


def test_gaussian():
    n_observations = 100
    x = np.arange(n_observations)
    y = make_data(
        Gaussian(
            centre=50.0,
            intensity=25.0,
            sigma=10.0
        ),
        x
    )

    prior_model = af.PriorModel(
        Gaussian,
        centre=af.GaussianPrior(
            mean=50,
            sigma=20
        ),
        intensity=af.GaussianPrior(
            mean=25,
            sigma=10
        ),
        sigma=af.GaussianPrior(
            mean=10,
            sigma=10
        )
    )

    def likelihood_function(instance):
        y_model = instance(x)
        return np.mean(
            _likelihood(
                y_model,
                y
            )
        )

    factor_model = ep.LikelihoodModel(
        prior_model,
        likelihood_function=likelihood_function
    )
    mean_field_approximation = factor_model.mean_field_approximation

    opt = ep.optimise.LaplaceOptimiser(
        mean_field_approximation,
        n_iter=3
    )

    opt.run()

    prior_value_dict = dict()
    for variable in factor_model.prior_variables:
        name = prior_model.path_for_prior(
            variable.prior
        )[0]
        prior_value_dict[name] = opt.model_approx[variable].mu

    assert prior_value_dict["centre"] == pytest.approx(50, rel=0.1)
    assert prior_value_dict["intensity"] == pytest.approx(25, rel=0.1)
    assert prior_value_dict["sigma"] == pytest.approx(10, rel=0.1)


@pytest.fixture(
    name="prior_model"
)
def make_prior_model():
    return af.PriorModel(
        Gaussian
    )


@pytest.fixture(
    name="likelihood_model"
)
def make_factor_model(
        prior_model
):
    def likelihood_function(
            z
    ):
        return 1

    return ep.LikelihoodModel(
        prior_model,
        likelihood_function=likelihood_function
    )


def test_messages(
        likelihood_model
):
    assert len(likelihood_model.message_dict) == 3


def test_graph(
        likelihood_model
):
    graph = likelihood_model.graph
    assert len(graph.factors) == 4


def test_prior_model_node(
        likelihood_model
):
    prior_model_node = likelihood_model.graph

    result = prior_model_node({
        variable: np.array([0.5])
        for variable
        in prior_model_node.variables
    })

    assert isinstance(
        result,
        ep.FactorValue
    )
