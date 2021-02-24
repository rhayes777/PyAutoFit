import numpy as np
import pytest

import autofit as af
from autofit import graphical as g

x = np.arange(100)
n = 10


@pytest.fixture(
    name="centre_model"
)
def make_centre_model():
    return af.PriorModel(
        af.GaussianPrior,
        mean=af.GaussianPrior(
            mean=50,
            sigma=10
        ),
        sigma=af.GaussianPrior(
            mean=10,
            sigma=5
        )
    )


def test_embedded_priors(
        centre_model
):
    assert isinstance(
        centre_model.random_instance().value_for(0.5),
        float
    )


def test_hierarchical_factor(
        centre_model
):
    factor = g.HierarchicalFactor(
        centre_model,
        af.GaussianPrior(50, 10)
    )
    laplace = g.LaplaceFactorOptimiser()

    collection = factor.optimise(laplace)
    print(collection)


@pytest.fixture(
    name="data"
)
def generate_data(
        centre_model
):
    data = []
    for _ in range(n):
        instance = centre_model.random_instance()
        data.append(
            np.array(list(map(
                instance,
                x
            )))
        )
    return data


def test_generate_data(
        data
):
    print(data)

# class Analysis:
#     def __init__(self, x, y, sigma=.04):
#         self.x = x
#         self.y = y
#         self.sigma = sigma
#
#     def log_likelihood_function(self, instance: Gaussian) -> np.array:
#         """
#         This function takes an instance created by the PriorModel and computes the
#         likelihood that it fits the data.
#         """
#         y_model = instance(self.x)
#         return np.sum(_likelihood(y_model, self.y) / self.sigma**2)
