from abc import ABC
from typing import Optional

from autofit.graphical.expectation_propagation import AbstractFactorOptimiser
from autofit.graphical.factor_graphs.factor import FactorKW
from autofit.mapper.prior_model.prior_model import PriorModel, AbstractPriorModel
from autofit.text.formatter import TextFormatter
from autofit.tools.namer import namer
from ..abstract import AbstractDeclarativeFactor


class AbstractModelFactor(FactorKW, AbstractDeclarativeFactor, ABC):
    @property
    def prior_model(self):
        return self._prior_model

    def __init__(
            self,
            prior_model: AbstractPriorModel,
            factor,
            optimiser: Optional[AbstractFactorOptimiser],
            prior_variable_dict,
            name=None,
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
            factor, **prior_variable_dict, name=name or namer(self.__class__.__name__)
        )

    @property
    def info(self) -> str:
        """
        Info describing this factor. Same as model info with the factor name.

        Output as part of graph.info
        """
        return f"{self.name}\n\n{self.prior_model.info}"

    def make_results_text(self, model_approx) -> str:
        """
        Create a string describing the posterior values after this factor
        during or after an EPOptimisation.

        Parameters
        ----------
        model_approx: EPMeanField

        Returns
        -------
        A string containing the name of this factor with the names and
        values of each associated variable in the mean field.
        """
        arguments = {
            prior: model_approx.mean_field[prior] for prior in self.prior_model.priors
        }
        updated_model = self.prior_model.gaussian_prior_model_for_arguments(arguments)

        formatter = TextFormatter()

        for path, prior in updated_model.path_priors_tuples:
            formatter.add(path, prior.mean)
        return f"{self.name}\n\n{formatter.text}"

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
        return super().optimise(optimiser, **kwargs).model[0]
