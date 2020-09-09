from typing import Callable

import autofit as af
from autofit import expectation_propagation as ep


class ModelFactor(ep.Factor):
    def __init__(
            self,
            prior_model: af.PriorModel,
            likelihood_function: Callable,
            prior_variables
    ):
        prior_variable_dict = dict()
        for prior_variable in prior_variables:
            prior_variable_dict[
                prior_variable.name
            ] = prior_variable

        def _factor(**kwargs):
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
            return likelihood_function(instance)

        super().__init__(
            _factor,
            **prior_variable_dict
        )
        self.likelihood_function = likelihood_function
        self.prior_model = prior_model


class PriorVariable(ep.Variable):
    def __init__(
            self,
            name: str,
            prior: af.Prior
    ):
        super().__init__(name)
        self.prior = prior
