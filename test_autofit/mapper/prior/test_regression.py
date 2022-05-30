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


@pytest.fixture(
    name="message"
)
def make_message(prior):
    return prior.message


def test_copy_limits(message):
    copied = message.copy()
    assert message.lower_limit == copied.lower_limit
    assert message.upper_limit == copied.upper_limit


def test_multiply_limits(message):
    multiplied = message * message
    assert message.lower_limit == multiplied.lower_limit
    assert message.upper_limit == multiplied.upper_limit

    multiplied = 1 * message
    assert message.lower_limit == multiplied.lower_limit
    assert message.upper_limit == multiplied.upper_limit


def test_sum_from_arguments(prior):
    added = prior + prior
    new = added.gaussian_prior_model_for_arguments({
        prior: af.UniformPrior(
            lower_limit=10,
            upper_limit=20,
        )
    })
    assert new.instance_from_prior_medians() == 30
