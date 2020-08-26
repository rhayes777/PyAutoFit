import numpy as np

import autofit as af
import autofit.expectation_propagation as ep
from .model import Gaussian, make_data


def test_model_factor():
    def image_function(
            instance
    ):
        return make_data(
            gaussian=instance,
            x=np.zeros(100)
        )

    model_factor = ep.ModelFactor(
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
    model = ep.MessagePassingPriorModel(
        prior_model,
        make_data
    )

    assert len(model.variables) == 3
    assert len(model.priors) == 3
