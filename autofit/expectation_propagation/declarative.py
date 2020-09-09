from typing import Callable
from typing import List
import numpy as np

from autofit.expectation_propagation.factor_graphs.factor import Factor, Variable
from autofit.expectation_propagation.mean_field import MeanFieldApproximation
from autofit.expectation_propagation.messages import NormalMessage
from autofit.mapper.prior_model.prior_model import Prior, PriorModel


class ModelFactor(Factor):
    def __init__(
            self,
            prior_model: PriorModel,
            likelihood_function: Callable,
            prior_variables
    ):
        """
        A factor in the graph that actually computes the likelihood of a model
        given values for each variable that model contains

        Parameters
        ----------
        prior_model
            A model with some dimensionality
        likelihood_function
            A function that evaluates how well an instance of the model fits some data
        prior_variables
            A collection of variables created by a larger model relevant to this model
        """
        prior_variable_dict = dict()
        for prior_variable in prior_variables:
            prior_variable_dict[
                prior_variable.name
            ] = prior_variable

        def _factor(
                **kwargs: np.ndarray
        ) -> int:
            arguments = dict()
            for name, array in kwargs.items():
                prior_id = int(name.split("_")[1])
                prior = prior_model.prior_with_id(
                    prior_id
                )
                arguments[prior] = array
            instance = prior_model.instance_for_arguments(
                arguments
            )
            return likelihood_function(instance)

        super().__init__(
            _factor,
            **prior_variable_dict
        )
        self.likelihood_function = likelihood_function
        self.prior_model = prior_model


class PriorVariable(Variable):
    def __init__(
            self,
            name: str,
            prior: Prior
    ):
        super().__init__(name)
        self.prior = prior


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
            PriorVariable(
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
            Factor(
                variable.prior,
                x=variable
            )
            for variable
            in self._prior_variables
        ]

    @property
    def message_dict(self):
        return {
            variable: NormalMessage.from_prior(
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
        return ModelFactor(
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
        return MeanFieldApproximation.from_kws(
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
