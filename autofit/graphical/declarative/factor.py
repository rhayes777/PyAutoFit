from abc import ABC
from typing import Optional

import numpy as np

from autofit import ModelInstance, Analysis
from autofit.graphical.expectation_propagation import AbstractFactorOptimiser
from autofit.graphical.factor_graphs.factor import Factor
from autofit.mapper.prior_model.prior_model import PriorModel, AbstractPriorModel
from .abstract import AbstractDeclarativeFactor


class AbstractModelFactor(Factor, AbstractDeclarativeFactor, ABC):
    @property
    def prior_model(self):
        return self._prior_model

    @property
    def optimiser(self):
        return self._optimiser

    def __init__(
            self,
            prior_model: AbstractPriorModel,
            factor,
            optimiser: Optional[AbstractFactorOptimiser],
            prior_variable_dict
    ):
        """
        A factor in the graph that actually computes the likelihood of a model
        given values for each variable that model contains

        Parameters
        ----------
        prior_model
            A model with some dimensionality
        optimiser
            A custom optimiser that will be used to fit this factor specifically
            instead of the default optimiser
        """
        self._prior_model = prior_model
        self._optimiser = optimiser

        super().__init__(
            factor,
            **prior_variable_dict
        )

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


class HierarchicalFactor(AbstractModelFactor):
    def __init__(
            self,
            prior_model,
            argument_prior,
            optimiser=None,
    ):
        def _factor(
                **kwargs
        ):
            argument = kwargs.pop(
                "argument"
            )
            arguments = dict()
            for name, array in kwargs.items():
                prior_id = int(name.split("_")[1])
                prior = prior_model.prior_with_id(
                    prior_id
                )
                arguments[prior] = array
            return prior_model.instance_for_arguments(
                arguments
            )(argument)

        prior_variable_dict = {
            prior.name: prior
            for prior
            in prior_model.priors
        }

        prior_variable_dict[
            "argument"
        ] = argument_prior

        super().__init__(
            prior_model=prior_model,
            factor=_factor,
            optimiser=optimiser,
            prior_variable_dict=prior_variable_dict
        )

    def log_likelihood_function(self, instance):
        return instance


class AnalysisFactor(AbstractModelFactor):
    @property
    def prior_model(self):
        return self._prior_model

    @property
    def optimiser(self):
        return self._optimiser

    def __init__(
            self,
            prior_model: AbstractPriorModel,
            analysis: Analysis,
            optimiser: Optional[AbstractFactorOptimiser] = None
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
        optimiser
            A custom optimiser that will be used to fit this factor specifically
            instead of the default optimiser
        """
        self.analysis = analysis

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

        prior_variable_dict = {
            prior.name: prior
            for prior
            in prior_model.priors
        }

        super().__init__(
            prior_model=prior_model,
            factor=_factor,
            optimiser=optimiser,
            prior_variable_dict=prior_variable_dict
        )

    def log_likelihood_function(
            self,
            instance: ModelInstance
    ) -> float:
        return self.analysis.log_likelihood_function(instance)

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
        )
