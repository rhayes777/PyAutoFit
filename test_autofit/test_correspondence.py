import pytest

from autofit.messages import UniformNormalMessage, NormalMessage
from autofit.messages.transform import phi_transform

OldUniformNormalMessage = NormalMessage.transformed(
    phi_transform, "UniformNormalMessage"
)


@pytest.fixture(name="message")
def make_message():
    return UniformNormalMessage(1.0, 0.5)


@pytest.fixture(name="x")
def make_x(message):
    return message.sample()


@pytest.fixture(name="old_message")
def make_old_message():
    return OldUniformNormalMessage(1.0, 0.5)


def test_logpdf_gradient(message, old_message, x):
    assert message.logpdf_gradient(x) == old_message.logpdf_gradient(x)


def test_log_pdf(message, old_message, x):
    assert message.logpdf(x) == old_message.logpdf(x)


def test_numerical_logpdf_gradient(message, old_message, x):
    assert message.numerical_logpdf_gradient(
        x
    ) == old_message.numerical_logpdf_gradient(x)


def test_logpdf_gradient_hessian(message, old_message, x):
    assert message.logpdf_gradient_hessian(x) == old_message.logpdf_gradient_hessian(x)


def test_factor(message, old_message, x):
    assert message.factor(x) == old_message.factor(x)
