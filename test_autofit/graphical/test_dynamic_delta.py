import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(name="prior")
def make_prior():
    return af.GaussianPrior(mean=1.0, sigma=2.0)


@pytest.fixture(name="numeric_mean_field")
def make_numeric_mean_field(prior):
    return g.MeanField({
        prior: 1.0
    })


def test_power(prior, numeric_mean_field):
    mean_field = g.MeanField({
        prior: prior.message
    })
    new = mean_field ** numeric_mean_field
    assert new[prior].sigma == 2.0
    assert new[prior].mean == 1.0


def test_add(prior, numeric_mean_field):
    new = 3.0 - numeric_mean_field
    assert new[prior] == 2.0
