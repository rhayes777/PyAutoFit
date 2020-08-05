from typing import Dict

from .factor_graphs import Variable, Factor
from .mean_field import MeanFieldApproximation
from .messages import NormalMessage, AbstractMessage


class MeanFieldPriorModel:
    def __init__(self, model, **priors):
        self.message_dict = dict()
        self.model = model
        self.prior_variables = list()

        for name, prior in priors.items():
            variable = getattr(
                model,
                name
            )
            self.prior_variables.append(
                variable
            )
            prior_factor = Factor(
                prior,
                x=variable
            )
            self.model *= prior_factor
            self.message_dict[variable] = NormalMessage.from_prior(
                prior
            )

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
