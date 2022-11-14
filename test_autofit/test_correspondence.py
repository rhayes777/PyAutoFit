import pytest

from autofit import UniformPrior
from autofit.messages import NormalMessage, UniformNormalMessage
from autofit.messages.composed_transform import TransformedMessage
from autofit.messages.transform import phi_transform, LinearShiftTransform

OldUniformNormalMessage = NormalMessage.transformed(
    phi_transform, "UniformNormalMessage"
)


@pytest.fixture(name="message")
def make_message():
    return UniformPrior(lower_limit=10, upper_limit=30).message


@pytest.fixture(name="x")
def make_x(message):
    return 0.5


@pytest.fixture(name="old_message")
def make_old_message():
    return OldUniformNormalMessage.shifted(shift=10, scale=20)(0, 1)


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


def test_variance(message, old_message):
    assert message.variance == old_message.variance


def test_value_for(message, old_message, x):
    assert message.value_for(x) == old_message.value_for(x)


def test_transform_variance():
    normal = NormalMessage(0, 1)

    shifted = TransformedMessage(normal, LinearShiftTransform(shift=10, scale=20),)
    reverted = TransformedMessage(
        shifted, LinearShiftTransform(shift=-10, scale=1 / 20),
    )

    assert shifted.variance == 400

    assert shifted.variance != normal.variance
    assert reverted.variance == pytest.approx(normal.variance)


def test_transform_uniform():
    normal = NormalMessage(0, 1)

    print(TransformedMessage(normal, phi_transform,).variance)

    transformed = TransformedMessage(
        normal, phi_transform, LinearShiftTransform(shift=10, scale=20),
    )
    print(transformed.variance)
    assert transformed.variance


# from_natural_parameters
# check_support
# project
# invert_natural_parameters
# cdf
# invert_sufficient_statistics
# _sample
# from_mode
# update_invalid
# log_base_measure
