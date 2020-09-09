from abc import ABC, abstractmethod
from typing import Callable, cast, Set, List, Dict

import numpy as np

from autofit import Prior
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.graphical.mean_field import MeanFieldApproximation
from autofit.graphical.messages import NormalMessage
from autofit.mapper.prior_model.prior_model import PriorModel


class AbstractModelFactor(ABC):
    @property
    @abstractmethod
    def model_factors(self) -> List["ModelFactor"]:
        """
        A list of factors that comprise a PriorModel and corresponding fitness function
        """

    @property
    def priors(self) -> Set[Prior]:
        """
        A set of all priors encompassed by the contained likelihood models
        """
        return {
            prior
            for model
            in self.model_factors
            for prior
            in model.prior_model.priors
        }

    @property
    def prior_factors(self) -> List[Factor]:
        """
        A list of factors that act as priors on latent variables. One factor exists
        for each unique prior.
        """
        return [
            Factor(
                prior,
                x=prior
            )
            for prior
            in self.priors
        ]

    @property
    def message_dict(self) -> Dict[Prior, NormalMessage]:
        """
        Dictionary mapping priors to messages.

        TODO: should support more than just GaussianPriors/NormalMessages
        """
        return {
            prior: NormalMessage.from_prior(
                prior
            )
            for prior
            in self.priors
        }

    @property
    def graph(self) -> FactorGraph:
        """
        The complete graph made by combining all factors and priors
        """
        return cast(
            FactorGraph,
            np.prod(
                [
                    model
                    for model
                    in self.model_factors
                ] + self.prior_factors
            )
        )

    def mean_field_approximation(self):
        return MeanFieldApproximation.from_kws(
            self.graph,
            self.message_dict
        )


class ModelFactor(Factor, AbstractModelFactor):
    def __init__(
            self,
            prior_model: PriorModel,
            likelihood_function: Callable
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
        """
        prior_variable_dict = {
            prior.name: prior
            for prior
            in prior_model.priors
        }

        def _factor(
                **kwargs: np.ndarray
        ) -> float:
            """
            Creates an instance of the prior model and evaluates it, forming
            a factor.

            Parameters
            ----------
            kwargs
                Arguments with names that are unique for each prior.

            Returns
            -------
            Calculated likelihood
            """
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

    def __mul__(self, other):
        """
        When two factors are multiplied together this creates a graph
        """
        return LikelihoodModelCollection([self]) * other

    @property
    def model_factors(self):
        return [self]


class LikelihoodModelCollection(FactorGraph, AbstractModelFactor):
    @property
    def model_factors(self):
        return self.factors
