from typing import Dict

from .factor_graphs import Variable, Factor
from .mean_field import MeanFieldApproximation
from .messages import NormalMessage, AbstractMessage


class VariableGroup:
    def __init__(self, variable, prior):
        self.variable = variable
        self.factor = Factor(
            prior,
            x=variable
        )
        self.message = NormalMessage.from_prior(
            prior
        )


class MeanFieldPriorModel:
    def __init__(
            self,
            model,
            **kwargs
    ):
        self._model = model
        self._variable_groups = dict()

        for name, item in kwargs.items():
            if isinstance(item, VariableGroup):
                group = item
            else:
                variable = getattr(
                    model,
                    name
                )
                group = VariableGroup(
                    variable,
                    item
                )
            self._variable_groups[
                name
            ] = group

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
