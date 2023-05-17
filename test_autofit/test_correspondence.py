import numpy as np
import pytest
from scipy.optimize import approx_fprime

from autofit import UniformPrior
from autofit.messages import NormalMessage
from autofit.messages.composed_transform import TransformedMessage
from autofit.messages.transform import phi_transform, LinearShiftTransform


@pytest.fixture(name="message")
def make_message():
    return UniformPrior(lower_limit=10.0, upper_limit=30.0).message


@pytest.fixture(name="x")
def make_x(message):
    return 0.5


def test_logpdf_gradient(message):
    x = 15
    a, b = message.logpdf_gradient(x)
    assert a == pytest.approx(-1.146406744764459)
    assert float(b) == pytest.approx(0.10612641)


def test_log_pdf(message):
    x = 15
    assert message.logpdf(x) == pytest.approx(-1.146406744764459)


def test_logpdf_gradient_hessian(message):
    x = 15
    answer = (-1.1464067447644593, 0.106126394117112, -0.03574918139293004)
    print(message.logpdf_gradient_hessian(x))
    for v1, v2 in zip(
        message.logpdf_gradient_hessian(x), answer
    ):
        assert v1 == pytest.approx(v2, abs=1e-3)


def test_calc_log_base_measure(message, x):
    assert message.calc_log_base_measure(x) == pytest.approx(-0.9189385332046727)


def test_to_canonical_form(message):
    x = 15
    assert np.allclose(
        message.to_canonical_form(x), np.array([-0.67448975, 0.45493642])
    )


def test_factor(message):
    x = 15
    assert message.factor(x) == -4.14213901831845


def test_value_for(message, x):
    assert message.value_for(x) == 20


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

    transformed = TransformedMessage(
        normal, phi_transform, LinearShiftTransform(shift=10.0, scale=20.0),
    )
    assert transformed.variance


def test_from_mode():
    message = UniformPrior(lower_limit=10, upper_limit=20).message
    mean = message.from_mode(14.03, covariance=np.zeros(())).mean
    assert mean == 14.03


def test_numerical_logpdf_gradient(message):
    logl, _ = message.numerical_logpdf_gradient(15)
    assert logl == pytest.approx(-1.1464067)


def test_log_pdf_gradient(message):
    logl, _ = message.logpdf_gradient(15)
    assert logl == pytest.approx(-1.1464067)


def test_regression():
    message = UniformPrior(lower_limit=0.3, upper_limit=1.1).message
    x = 1.076

    log_likelihood, gradient = message.logpdf_gradient(x)
    numerical_log_likelihood, numerical_gradient = message.numerical_logpdf_gradient(x)
    approx_gradient = approx_fprime(x, message.logpdf, epsilon=1e-8)

    assert log_likelihood == numerical_log_likelihood
    assert float(gradient) == pytest.approx(numerical_gradient, rel=0.001)
    assert float(approx_gradient) == pytest.approx(numerical_gradient, rel=0.001)
