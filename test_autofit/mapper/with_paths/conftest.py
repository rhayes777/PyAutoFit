import pytest

import autofit as af


@pytest.fixture(
    name="gaussian_1"
)
def make_gaussian_1():
    return af.Model(
        af.Gaussian
    )


@pytest.fixture(
    name="model"
)
def make_model(gaussian_1):
    return af.Collection(
        gaussian_1=gaussian_1,
        gaussian_2=af.Model(
            af.Gaussian
        ),
    )
