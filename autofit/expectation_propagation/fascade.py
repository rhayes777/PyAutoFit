from abc import ABC, abstractmethod
from typing import Dict, Callable, List

from .factor_graphs import Variable, Factor, FactorGraph, AbstractNode
from .mean_field import MeanFieldApproximation
from .messages import NormalMessage, AbstractMessage


class PriorVariable:
    def __init__(
            self,
            prior: Callable,
            variable: Variable
    ):
        """
        Comprises a prior and the variable to which it corresponds

        Parameters
        ----------
        prior
            A prior on the variable
        variable
            A placeholder for a variable
        """
        self.variable = variable
        self.factor = Factor(
            prior,
            x=variable
        )
        self.message = NormalMessage.from_prior(
            prior
        )


class AbstractMeanFieldPriorModel(ABC):
    def mean_field_approximation(
            self,
            variable_message_dict: Dict[Variable, AbstractMessage]
    ) -> MeanFieldApproximation:
        """
        Create a mean field approximation from this model.

        Parameters
        ----------
        variable_message_dict
            A dictionary mapping variables to messages. This is so messages corresponding
            to data can be passed in.

        Returns
        -------
        A mean-field approximation of this model
        """
        return MeanFieldApproximation.from_kws(
            self.model,
            {
                **self.message_dict,
                **variable_message_dict
            }
        )

    def __mul__(
            self,
            other: "AbstractMeanFieldPriorModel"
    ) -> "CompoundMeanFieldPriorModel":
        """
        Combine this model with another model

        Parameters
        ----------
        other
            Another model

        Returns
        -------
        A combined model
        """
        return CompoundMeanFieldPriorModel(
            self, other
        )

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def message_dict(self):
        pass

    @property
    @abstractmethod
    def prior_variables(self):
        pass


class CompoundMeanFieldPriorModel(AbstractMeanFieldPriorModel):
    """
    A combination of two or more models
    """

    @property
    def model(self) -> FactorGraph:
        """
        Combine the factor graphs of the underlying models
        """
        model = self.mean_field_prior_models[0].model
        for mean_field_prior_model in self.mean_field_prior_models[1:]:
            model *= mean_field_prior_model.model
        return model

    @property
    def message_dict(self) -> Dict[Variable, AbstractMessage]:
        """
        Combine the message dictionaries of the underlying models
        """
        message_dict = dict()
        for mean_field_prior_model in self.mean_field_prior_models:
            message_dict.update(
                mean_field_prior_model.message_dict
            )
        return message_dict

    def __init__(self, *mean_field_prior_models):
        self.mean_field_prior_models = mean_field_prior_models

    @property
    def prior_variables(self) -> List[Variable]:
        """
        Combine the prior-variables of the underlying models
        """
        return [
            prior_variable
            for model
            in self.mean_field_prior_models
            for prior_variable
            in model.prior_variables
        ]


class MeanFieldPriorModel(AbstractMeanFieldPriorModel):
    def __init__(
            self,
            model: AbstractNode,
            **kwargs: Callable
    ):
        """
        Wraps a factor or graph

        Parameters
        ----------
        model
            A factor or graph
        kwargs
            Additional model components, such as priors
        """
        self._model = model
        self._variable_groups = dict()

        for name, item in kwargs.items():
            variable = getattr(
                model,
                name
            )
            group = PriorVariable(
                item,
                variable
            )
            self._variable_groups[
                name
            ] = group

    @property
    def model(self) -> FactorGraph:
        """
        Create a factor graph by combining the underlying model with any additional factors
        """
        model = self._model
        for group in self._variable_groups.values():
            model *= group.factor
        return model

    @property
    def message_dict(self) -> Dict[Variable, AbstractMessage]:
        """
        Create a message dict from all the underlying messages
        """
        return {
            group.variable: group.message
            for group
            in self._variable_groups.values()
        }

    @property
    def prior_variables(self) -> List[Variable]:
        """
        A list of variables contained in this model
        """
        return [
            group.variable
            for group
            in self._variable_groups.values()
        ]
