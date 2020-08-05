from typing import Dict

from .factor_graphs import Variable, Factor
from .mean_field import MeanFieldApproximation
from .messages import NormalMessage, AbstractMessage


class VariableGroup:
    def __init__(self, variable, factor, message):
        self.variable = variable
        self.factor = factor
        self.message = message


class MeanFieldPriorModel:
    def __init__(
            self,
            model,
            **priors
    ):
        self._model = model
        self._variable_groups = dict()

        for name, prior in priors.items():
            variable = getattr(
                model,
                name
            )
            factor = Factor(
                prior,
                x=variable
            )
            message = NormalMessage.from_prior(
                prior
            )
            self._variable_groups[
                name
            ] = VariableGroup(
                variable,
                factor,
                message
            )

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
