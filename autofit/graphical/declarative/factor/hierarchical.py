from typing import Set

from autofit import Prior
from .abstract import AbstractModelFactor


class HierarchicalFactor(AbstractModelFactor):
    def __init__(
            self,
            prior_model,
            argument_prior,
            optimiser=None,
            name=None
    ):
        self.argument_prior = argument_prior

        def _factor(
                **kwargs
        ):
            argument = kwargs.pop(
                "argument"
            )
            arguments = dict()
            for name_, array in kwargs.items():
                prior_id = int(name_.split("_")[1])
                prior = prior_model.prior_with_id(
                    prior_id
                )
                arguments[prior] = array
            result = prior_model.instance_for_arguments(
                arguments
            )(argument)
            return result

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
            prior_variable_dict=prior_variable_dict,
            name=name
        )

    def log_likelihood_function(self, instance):
        return instance

    @property
    def priors(self) -> Set[Prior]:
        priors = super().priors
        priors.add(
            self.argument_prior
        )
        return priors
