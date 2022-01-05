import numpy as np
import pytest

import autofit as af
from test_autofit.graphical.gaussian.model import Gaussian, make_data


@pytest.fixture(
    name="x"
)
def make_x():
    return np.arange(100)


@pytest.fixture(
    name="y"
)
def make_y(x):
    return make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)


@pytest.fixture(
    name="prior_model"
)
def make_prior_model():
    return af.PriorModel(
        Gaussian,
        centre=af.GaussianPrior(mean=50, sigma=20),
        normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )
