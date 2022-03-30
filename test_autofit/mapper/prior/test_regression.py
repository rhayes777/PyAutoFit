import pytest

import autofit as af


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.GaussianPrior(
        mean=1,
        sigma=2,
        lower_limit=3,
        upper_limit=4
    )


def test_copy_limits(prior):
    copied = prior.copy()
    assert prior.lower_limit == copied.lower_limit
    assert prior.upper_limit == copied.upper_limit


def test_multiply_limits(prior):
    multiplied = prior * prior
    assert prior.lower_limit == multiplied.lower_limit
    assert prior.upper_limit == multiplied.upper_limit

    multiplied = 1 * prior
    assert prior.lower_limit == multiplied.lower_limit
    assert prior.upper_limit == multiplied.upper_limit


def test_divide_prior(prior):
    divided = prior / prior
    assert prior.lower_limit == divided.lower_limit
    assert prior.upper_limit == divided.upper_limit
