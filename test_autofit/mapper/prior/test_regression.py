from copy import copy

import pytest

import autofit as af


@pytest.fixture(name="prior")
def make_prior():
    return af.GaussianPrior(mean=1, sigma=2, lower_limit=3, upper_limit=4)


@pytest.fixture(name="message")
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


@pytest.fixture(name="uniform_prior")
def make_uniform_prior():
    return af.UniformPrior(
        lower_limit=10,
        upper_limit=20,
    )


def test_sum_from_arguments(prior, uniform_prior):
    added = prior + prior
    new = added.gaussian_prior_model_for_arguments({prior: uniform_prior})
    assert new.instance_from_prior_medians() == 30


def test_negative_from_arguments(prior, uniform_prior):
    negative = -prior
    new = negative.gaussian_prior_model_for_arguments({prior: uniform_prior})
    assert new.instance_from_prior_medians() == -15


def test_sum_with_float(prior, uniform_prior):
    added = prior + 15
    new = added.gaussian_prior_model_for_arguments({prior: uniform_prior})
    assert new.instance_from_prior_medians() == 30


def test_name(prior):
    new = prior.new()

    assert new.id != prior.id
    assert new.name != prior.name
