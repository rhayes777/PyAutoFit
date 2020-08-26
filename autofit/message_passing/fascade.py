from abc import ABC, abstractmethod
from typing import Dict

from .factor_graphs import Variable, Factor
from .mean_field import MeanFieldApproximation
from .messages import NormalMessage, AbstractMessage


class PriorVariable:
    def __init__(self, prior, variable):
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
    ):
        return MeanFieldApproximation.from_kws(
            self.model,
            {
                **self.message_dict,
                **variable_message_dict
            }
        )

    def __mul__(self, other):
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
    @property
    def model(self):
        model = self.mean_field_prior_models[0].model
        for mean_field_prior_model in self.mean_field_prior_models[1:]:
            model *= mean_field_prior_model.model
        return model

    @property
    def message_dict(self):
        message_dict = dict()
        for mean_field_prior_model in self.mean_field_prior_models:
            message_dict.update(
                mean_field_prior_model.message_dict
            )
        return message_dict

    def __init__(self, *mean_field_prior_models):
        self.mean_field_prior_models = mean_field_prior_models

    @property
    def prior_variables(self):
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
            model,
            **kwargs
    ):
        self._model = model
        self._variable_groups = dict()

        for name, item in kwargs.items():
            if isinstance(item, PriorVariable):
                group = item
            else:
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

    def __getitem__(self, item):
        return self._variable_groups[item]

    def __setitem__(self, key, value):
        self._variable_groups[key] = value

    @property
    def variable_groups(self):
        return self._variable_groups

    @property
    def model(self):
        model = self._model
        for group in self._variable_groups.values():
            model *= group.factor
        return model

    @property
    def message_dict(self):
        return {
            group.variable: group.message
            for group
            in self._variable_groups.values()
        }

    @property
    def prior_variables(self):
        return [
            group.variable
            for group
            in self._variable_groups.values()
        ]
