from typing import Dict, Callable

import numpy as np

import autofit as af
from autofit import expectation_propagation as ep
from autofit.expectation_propagation.factor_graphs import FactorValue


class ModelFactor(ep.Factor):
    def __init__(
            self,
            prior_model: af.PriorModel,
            image_function: Callable,
            prior_variables
    ):
        prior_variable_dict = dict()
        for prior_variable in prior_variables:
            prior_variable_dict[
                prior_variable.name
            ] = prior_variable
        super().__init__(
            image_function,
            instance=ep.Variable("instance"),
            **prior_variable_dict
        )
        self.image_function = image_function
        self.prior_model = prior_model

    def __call__(
            self,
            variable_dict: Dict[ep.Variable, np.array]
    ) -> FactorValue:
        """
        Call the underlying factor

        Parameters
        ----------
        variable_dict

        Returns
        -------
        Object encapsulating the result of the function call
        """
        arguments = {
            variable.prior: array
            for variable, array
            in variable_dict.items()
            if isinstance(
                variable,
                PriorVariable
            )
        }
        instance = self.prior_model.instance_for_arguments(
            arguments
        )
        return self.image_function(
            instance
        )


class PriorVariable(ep.Variable):
    def __init__(
            self,
            name: str,
            prior: af.Prior
    ):
        super().__init__(name)
        self.prior = prior
