from typing import List

import numpy as np
import pytest

import autofit as af
import autofit.expectation_propagation as ep
from test_autofit.unit.expectation_propagation.gaussian.model import Gaussian, make_data, _likelihood


class FactorModel:
    def __init__(
            self,
            likelihood_models: List["LikelihoodModel"]
    ):
        self.likelihood_models = likelihood_models
        self._unique_priors = {
            prior: path
            for prior_model
            in self.prior_models
            for path, prior
            in prior_model.path_priors_tuples
        }
        self._prior_variables = [
            ep.declarative.PriorVariable(
                f"prior_{prior.id}",
                prior
            )
            for prior, path in self._unique_priors.items()
        ]
        self._prior_variable_map = {
            prior_variable.prior: prior_variable
            for prior_variable in self._prior_variables
        }

    @property
    def prior_variables(self):
        return self._prior_variables

    @property
    def prior_models(self):
        return [
            model.prior_model
            for model
            in self.likelihood_models
        ]

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
    def message_dict(self):
        return {
            variable: ep.NormalMessage.from_prior(
                variable.prior
            )
            for variable
            in self._prior_variables
        }

    def _node_for_likelihood_model(
            self,
            likelihood_model: "LikelihoodModel"
    ):
        prior_variables = [
            self._prior_variable_map[
                prior
            ]
            for prior
            in likelihood_model.prior_model.priors
        ]
        return ep.ModelFactor(
            likelihood_model.prior_model,
            likelihood_model.likelihood_function,
            prior_variables
        )

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

        graph = self._node_for_likelihood_model(
            self.likelihood_models[0],
        )
        for likelihood_model in self.likelihood_models[1:]:
            graph *= self._node_for_likelihood_model(
                likelihood_model
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

    def __mul__(self, other: "FactorModel"):
        return FactorModel(
            other.likelihood_models + self.likelihood_models
        )


class LikelihoodModel(FactorModel):
    def __init__(
            self,
            prior_model,
            likelihood_function
    ):
        self.prior_model = prior_model
        self.likelihood_function = likelihood_function
        super().__init__([self])


def test_gaussian():
    n_observations = 100
    x = np.arange(n_observations)
    y = make_data(
        Gaussian(
            centre=100.0,
            intensity=25.0,
            sigma=10.0
        ),
        x
    )

    prior_model = af.PriorModel(
        Gaussian,
        centre=af.GaussianPrior(
            mean=100,
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
        return _likelihood(
            make_data(
                instance,
                x
            ),
            y
        )

    factor_model = LikelihoodModel(
        prior_model,
        likelihood_function=likelihood_function
    )
    mean_field_approximation = factor_model.mean_field_approximation

    opt = ep.optimise.LaplaceOptimiser(
        mean_field_approximation,
        n_iter=9
    )

    opt.run()

    for variable in factor_model.prior_variables:
        print(f"{variable.name} = {opt.model_approx[variable].mu}")


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

    return LikelihoodModel(
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
