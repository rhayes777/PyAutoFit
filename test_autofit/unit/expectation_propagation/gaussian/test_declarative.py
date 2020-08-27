import numpy as np

import pytest

import autofit as af
import autofit.expectation_propagation as ep
from .model import Gaussian, make_data


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
        unique_priors = {
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
            for prior, path in unique_priors.items()
        ]

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
            prior_model
    ):
        z = ep.Variable("z")
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
        return graph


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
