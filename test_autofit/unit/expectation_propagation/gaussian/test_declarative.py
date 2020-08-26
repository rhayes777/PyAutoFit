import numpy as np

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
        self.prior_models = prior_models
        self.image_function = image_function
        self.likelihood_function = likelihood_function

    @property
    def graph(self):
        unique_priors = {
            prior: path
            for model
            in self.prior_models
            for path, prior
            in model.path_priors_tuples
        }
        prior_variables = [
            ep.declarative.PriorVariable(
                "_".join(
                    path
                ),
                prior
            )
            for prior, path in unique_priors.items()
        ]

        graph = ep.ModelFactor(
            self.prior_models[0],
            self.image_function,
            prior_variables
        )
        for prior_model in self.prior_models[1:]:
            graph *= ep.ModelFactor(
                prior_model,
                self.image_function,
                prior_variables
            )
        return graph


def test_factor_model():
    pass


def test_model_factor():
    def image_function(
            instance
    ):
        return make_data(
            gaussian=instance,
            x=np.zeros(100)
        )

    model_factor = FactorModel(
        af.PriorModel(
            Gaussian
        ),
        image_function=image_function,
        likelihood_function=None
    )
    graph = model_factor.graph

    result = graph({
        graph.centre: 1.0,
        graph.intensity: 0.5,
        graph.sigma: 0.5
    })

    assert isinstance(
        result,
        np.ndarray
    )
