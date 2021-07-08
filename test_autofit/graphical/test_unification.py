import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.GaussianPrior(
        mean=1,
        sigma=2
    )


@pytest.fixture(
    name="message"
)
def make_message(prior):
    return g.AbstractMessage.from_prior(
        prior
    )


def test():
    mean_field = g.MeanField({

    })

    print(mean_field.prior_count)
    mean_field.instance_for_arguments({})


def test_from_prior(
        prior,
        message
):
    assert message.id == prior.id
    assert message.mu == prior.mean
    assert message.sigma == prior.sigma


def test_retain_id(
        message
):
    new_message = message * message
    assert new_message.id == message.id


def test_bad_id(
        message
):
    new_message = message * message
    new_message.id = 2

    with pytest.raises(
        AssertionError
    ):
        new_message * message
