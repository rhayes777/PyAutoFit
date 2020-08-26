from typing import Dict, Callable

import numpy as np

import autofit as af
from autofit import expectation_propagation as ep
from autofit.expectation_propagation.factor_graphs import FactorValue
from .model import Gaussian, make_data


class ModelFactor(ep.Factor):
    def __init__(
            self,
            prior_model: af.PriorModel,
            image_function: Callable
    ):
        instance_variable_dict = dict()
        for path, prior in prior_model.path_priors_tuples:
            name = "_".join(path)
            instance_variable_dict[
                name
            ] = PriorVariable(
                name,
                prior
            )

        super().__init__(
            image_function,
            instance=ep.Variable("instance"),
            **instance_variable_dict
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


class MessagePassingPriorModel(ep.MeanFieldPriorModel):
    def __init__(
            self,
            prior_model,
            image_function
    ):
        super().__init__(
            ModelFactor(
                prior_model,
                image_function
            ),
            **dict(
                prior_model.prior_tuples
            )
        )

    @property
    def priors(self):
        return [
            prior_variable.factor
            for prior_variable
            in self.prior_variables
        ]


class PriorVariable(ep.Variable):
    def __init__(
            self,
            name: str,
            prior: af.Prior
    ):
        super().__init__(name)
        self.prior = prior


def test_model_factor():
    def image_function(
            instance
    ):
        return make_data(
            gaussian=instance,
            x=np.zeros(100)
        )

    model_factor = ModelFactor(
        af.PriorModel(
            Gaussian
        ),
        image_function
    )

    result = model_factor({
        model_factor.centre: 1.0,
        model_factor.intensity: 0.5,
        model_factor.sigma: 0.5
    })

    assert isinstance(
        result,
        np.ndarray
    )


def test_declarative_model():
    prior_model = af.PriorModel(
        Gaussian
    )
    model = MessagePassingPriorModel(
        prior_model,
        make_data
    )

    assert len(model.variables) == 3
    assert len(model.priors) == 3
