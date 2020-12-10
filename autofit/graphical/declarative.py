from abc import ABC, abstractmethod
from typing import Callable, cast, Set, List, Dict

import numpy as np

from autofit import ModelInstance, Analysis, Paths
from autofit.graphical.expectation_propagation import EPMeanField
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.graph import FactorGraph
from autofit.graphical.messages import NormalMessage
from autofit.mapper.prior.prior import Prior
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.mapper.prior_model.prior_model import PriorModel, AbstractPriorModel


class AbstractModelFactor(Analysis, ABC):
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
                cast(
                    Callable,
                    prior
                ),
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

    def mean_field_approximation(self) -> EPMeanField:
        """
        Returns a EPMeanField of the factor graph
        """
        return EPMeanField.from_approx_dists(
            self.graph,
            self.message_dict
        )

    def optimise(self, optimiser) -> CollectionPriorModel:
        """
        Use an EP Optimiser to optimise the graph associated with this collection
        of factors and create a Collection to represent the results.

        Parameters
        ----------
        optimiser
            An optimiser that acts on graphs

        Returns
        -------
        A collection of prior models
        """
        updated_model, status = optimiser.run(
            self.mean_field_approximation()
        )

        collection = CollectionPriorModel([
            factor.prior_model
            for factor
            in self.model_factors
        ])
        arguments = {
            prior: updated_model.mean_field[
                prior
            ].as_prior()
            for prior
            in collection.priors
        }

        return collection.gaussian_prior_model_for_arguments(
            arguments
        )

    def visualize(
            self,
            paths: Paths,
            instance: ModelInstance,
            during_analysis: bool
    ):
        """
        Visualise the instances provided using each factor.

        Instances in the ModelInstance must have the same order as the factors.

        Parameters
        ----------
        paths
            Object describing where data should be saved to
        instance
            A collection of instances, each corresponding to a factor
        during_analysis
            Is this visualisation during analysis?
        """
        for model_factor, instance in zip(
                self.model_factors,
                instance
        ):
            model_factor.visualize(
                paths,
                instance,
                during_analysis
            )

    def log_likelihood_function(
            self,
            instance: ModelInstance
    ) -> float:
        """
        Compute the combined likelihood of each factor from a collection of instances
        with the same ordering as the factors.

        Parameters
        ----------
        instance
            A collection of instances, one corresponding to each factor

        Returns
        -------
        The combined likelihood of all factors
        """
        likelihood = abs(
            self.model_factors[0].analysis.log_likelihood_function(
                instance[0]
            )
        )
        for model_factor, instance_ in zip(
                self.model_factors[1:],
                instance[1:]
        ):
            likelihood *= abs(
                model_factor.analysis.log_likelihood_function(
                    instance_
                )
            )
        return -likelihood

    @property
    def global_prior_model(self) -> CollectionPriorModel:
        """
        A collection of prior models, with one model for each factor.
        """
        return CollectionPriorModel([
            model_factor.prior_model
            for model_factor
            in self.model_factors
        ])


class ModelFactor(Factor, AbstractModelFactor):
    def __init__(
            self,
            prior_model: AbstractPriorModel,
            analysis: Analysis
    ):
        """
        A factor in the graph that actually computes the likelihood of a model
        given values for each variable that model contains

        Parameters
        ----------
        prior_model
            A model with some dimensionality
        analysis
            A class that implements a function which evaluates how well an
            instance of the model fits some data
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
        Returns an instance of the prior model and evaluates it, forming
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
            return analysis.log_likelihood_function(
                instance
            )

        super().__init__(
            _factor,
            **prior_variable_dict
        )
        self.prior_model = prior_model
        self.analysis = analysis

    @property
    def model_factors(self) -> List["ModelFactor"]:
        return [self]

    def optimise(self, optimiser) -> PriorModel:
        """
        Optimise this factor on its own returning a PriorModel
        representing the final state of the messages.

        Parameters
        ----------
        optimiser

        Returns
        -------
        A PriorModel representing the optimised factor
        """
        return super().optimise(
            optimiser
        )[0]


class GraphicalModel(AbstractModelFactor):
    def __init__(self, *model_factors: ModelFactor):
        """
        A collection of factors that describe models, which can be
        used to create a graph and messages.

        If the models have shared priors then the graph has shared variables

        Parameters
        ----------
        model_factors
        """
        self._model_factors = model_factors

    @property
    def model_factors(self):
        return self._model_factors
