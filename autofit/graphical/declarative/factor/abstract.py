from abc import ABC
from typing import Optional

from autofit.graphical.expectation_propagation import AbstractFactorOptimiser
from autofit.graphical.factor_graphs.factor import Factor
from autofit.mapper.prior_model.prior_model import PriorModel, AbstractPriorModel
from autofit.tools.namer import namer
from ..abstract import AbstractDeclarativeFactor


class AbstractModelFactor(Factor, AbstractDeclarativeFactor, ABC):
    @property
    def prior_model(self):
        return self._prior_model

    def __init__(
            self,
            prior_model: AbstractPriorModel,
            factor,
            optimiser: Optional[AbstractFactorOptimiser],
            prior_variable_dict,
            name=None
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
        self.optimiser = optimiser

        super().__init__(
            factor,
            **prior_variable_dict,
            name=name or namer(self.__class__.__name__)
        )

    def optimise(self, optimiser, **kwargs) -> PriorModel:
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
            optimiser, **kwargs
        )[0]
