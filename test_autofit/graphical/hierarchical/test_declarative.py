import numpy as np
import pytest

import autofit as af
from autofit.mock.mock import Gaussian
from ..gaussian.model import make_data

x = np.arange(100)


class Sampling(af.AbstractPriorModel):
    def __init__(
            self,
            model,
            method,
            **kwargs
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.kwargs = kwargs

    def _instance_for_arguments(self, arguments):
        # noinspection PyProtectedMember
        instance = self.model._instance_for_arguments(
            arguments
        )
        return self.method(
            instance,
            **{
                key: value.random()
                if isinstance(
                    value,
                    af.Prior
                )
                else value
                for key, value
                in self.kwargs.items()
            }
        )


@pytest.fixture(
    name="sampling"
)
def make_sampling():
    return Sampling(
        af.PriorModel(
            Gaussian,
            centre=25,
            sigma=15
        ),
        Gaussian.inverse,
        y=af.UniformPrior(
            lower_limit=0.0,
            upper_limit=1.0
        )
    )


def test(sampling):
    centres = [
        sampling.instance_from_prior_medians()
        for _ in range(10)
    ]
    gaussians = [
        Gaussian(
            centre=centre,
            intensity=1,
            sigma=10
        )
        for centre
        in centres
    ]
    datasets = [
        make_data(
            gaussian,
            x
        )
        for gaussian
        in gaussians    
    ]
    print(datasets)


def test_sample(sampling):
    assert isinstance(
        sampling.instance_from_prior_medians(),
        float
    )
