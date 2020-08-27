import numpy as np
import pytest

import autofit as af
import autofit.expectation_propagation as ep
from test_autofit.unit.expectation_propagation.gaussian.model import Gaussian, make_data, _likelihood


class FactorModel:
    def __init__(
            self,
            *prior_models: af.PriorModel,
            image_function,
            likelihood_function
    ):
        self._prior_models = prior_models
        self._image_function = image_function
        self._likelihood_function = likelihood_function
        self._unique_priors = {
            prior: path
            for model
            in self._prior_models
            for path, prior
            in model.path_priors_tuples
        }
        self._prior_variables = [
            ep.declarative.PriorVariable(
                "_".join(
                    path
                ),
                prior
            )
            for prior, path in self._unique_priors.items()
        ]

        self._deterministic_variables = {
            prior_model: ep.Variable(
                "z",
                ep.Plate(
                    "observations"
                )
            )
            for prior_model
            in prior_models
        }

    @property
    def prior_factors(self):
        return [
            ep.Factor(
                variable.prior,
                x=variable
            )
            for variable
            in self._prior_variables
        ]

    @property
    def shape(self):
        return self._image_function(
            self._prior_models[0].instance_from_prior_medians()
        ).shape

    @property
    def message_dict(self):
        return {
            **{
                variable: ep.NormalMessage.from_mode(
                    np.zeros(self.shape),
                    100
                )
                for variable
                in self._deterministic_variables.values()
            },
            **{
                variable: ep.NormalMessage.from_prior(
                    variable.prior
                )
                for variable
                in self._prior_variables
            }
        }

    def _node_for_prior_model(
            self,
            prior_model
    ):
        return ep.ModelFactor(
            prior_model,
            self._image_function,
            self._prior_variables
        )

    def _graph_for_prior_model(
            self,
            prior_model,
    ):
        z = self._deterministic_variables[
            prior_model
        ]
        likelihood_factor = ep.Factor(
            self._likelihood_function,
            z=z
        )
        model_factor = self._node_for_prior_model(
            prior_model
        ) == z
        return model_factor * likelihood_factor

    @property
    def graph(self):
        """
        - Test graph with associated fitness function
        - Test running an actual fit for a Gaussian
        - Test creating a graph with multiple gaussians and shared priors
        - Also need to generate a dictionary mapping each of the prior variables to an initial message
        - Can multiple instances of this class be combined? That would allow customisation of image and likelihood
        functions
        """

        graph = self._graph_for_prior_model(
            self._prior_models[0],
        )
        for prior_model in self._prior_models[1:]:
            graph *= self._graph_for_prior_model(
                prior_model
            )
        for prior_factor in self.prior_factors:
            graph *= prior_factor
        return graph

    @property
    def mean_field_approximation(self):
        return ep.MeanFieldApproximation.from_kws(
            self.graph,
            self.message_dict
        )


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
            mean=20,
            sigma=10
        ),
        sigma=af.GaussianPrior(
            mean=20,
            sigma=10
        )
    )

    def image_function(
            instance
    ):
        return make_data(
            instance,
            x
        )

    def likelihood_function(z):
        return _likelihood(z, y)

    factor_model = FactorModel(
        prior_model,
        image_function=image_function,
        likelihood_function=likelihood_function
    )
    mean_field_approximation = factor_model.mean_field_approximation

    opt = ep.optimise.LaplaceOptimiser(
        mean_field_approximation,
        n_iter=3
    )

    opt.run()

    # for variable in prior_model.variables:
    #     print(f"{variable.name} = {opt.model_approx[variable].mu}")


@pytest.fixture(
    name="prior_model"
)
def make_prior_model():
    return af.PriorModel(
        Gaussian
    )


@pytest.fixture(
    name="factor_model"
)
def make_factor_model(
        prior_model
):
    def image_function(
            instance
    ):
        return make_data(
            gaussian=instance,
            x=np.zeros(100)
        )

    def likelihood_function(
            z
    ):
        return 1

    return FactorModel(
        prior_model,
        image_function=image_function,
        likelihood_function=likelihood_function
    )


def test_messages(
        factor_model
):
    assert len(factor_model.message_dict) == 4


def test_graph(
        factor_model
):
    graph = factor_model.graph
    assert len(graph.factors) == 5


def test_prior_model_node(
        prior_model,
        factor_model
):
    prior_model_node = factor_model._node_for_prior_model(
        prior_model
    )

    result = prior_model_node({
        prior_model_node.centre: 1.0,
        prior_model_node.intensity: 0.5,
        prior_model_node.sigma: 0.5
    })

    assert isinstance(
        result,
        np.ndarray
    )
