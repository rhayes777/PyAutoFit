import numpy as np
import pytest

import autofit as af
from autofit import graphical as g
from autofit.mock.mock import Gaussian
# noinspection PyUnresolvedReferences
from ..gaussian.model import make_data, _likelihood, Analysis

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
            centre=0,
            sigma=2,
            intensity=10
        ),
        Gaussian.inverse,
        y=af.UniformPrior(
            lower_limit=-1.0,
            upper_limit=1.0
        )
    )


@pytest.mark.parametrize(
    "x",
    [
        0.1, 0.2, 0.3
    ]
)
def test_inverse(x):
    gaussian = Gaussian()
    assert gaussian.inverse(
        gaussian(x)
    ) == x


class CentreAnalysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return instance


def _test(sampling):
    parent_centre = af.GaussianPrior(
        mean=50,
        sigma=10,
    )
    parent_sigma = af.GaussianPrior(
        mean=10,
        sigma=10
    )

    model = g.FactorGraphModel()

    for i in range(10):
        centre = sampling.instance_from_prior_medians()
        gaussian = Gaussian(
            centre=centre,
            intensity=25,
            sigma=10
        )

        y = gaussian(x)

        print(y)

        centre = af.GaussianPrior(mean=50, sigma=20)

        prior_model = af.PriorModel(
            Gaussian,
            centre=centre,
            intensity=25,
            sigma=10,
        )

        model.add(
            g.ModelFactor(
                prior_model,
                analysis=Analysis(
                    x=x,
                    y=y
                )
            )
        )

        model.add(
            g.ModelFactor(
                Sampling(
                    af.PriorModel(
                        Gaussian,
                        centre=parent_centre,
                        sigma=parent_sigma,
                        intensity=25
                    ),
                    Gaussian.__call__,
                    xvalues=centre
                ),
                analysis=CentreAnalysis()
            )
        )

    laplace = g.LaplaceFactorOptimiser()

    collection = model.optimise(laplace)
    print(collection)


def test_sample(sampling):
    assert isinstance(
        sampling.instance_from_prior_medians(),
        float
    )
