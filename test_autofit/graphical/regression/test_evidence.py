import pytest

import autofit as af
from autofit import messages as m


@pytest.mark.parametrize(
    "message",
    [
        af.UniformPrior().message,
        af.GaussianPrior(
            mean=1.0,
            sigma=2.0
        ).message,
        m.NormalMessage(
            mean=0.0,
            sigma=1.0
        )
    ]
)
def test_log_normalisation(message):
    assert message.log_normalisation() != 0.0


@pytest.fixture(
    name="message"
)
def make_message():
    return m.NormalMessage(
        mean=0.0,
        sigma=1.0,
    )


def test_multi_evidence(message):
    assert message.log_normalisation(message) != 0.0


def test_identity(message):
    assert message == message
    assert hash(message) == hash(message)
    assert {message, message} == {message}


def test_hash_transformed():
    message = af.UniformPrior().message
    assert isinstance(
        hash(message),
        int
    )
