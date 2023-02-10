import pytest

import autofit as af


@pytest.fixture(
    name="model"
)
def make_model():
    model = af.Model(
        af.Gaussian
    )
    return model


def test_sort_alphabetically(
        model
):
    assert model.sort_priors_alphabetically([
        model.sigma,
        model.centre,
        model.normalization,
    ]) == [
               model.centre,
               model.normalization,
               model.sigma,
           ]
