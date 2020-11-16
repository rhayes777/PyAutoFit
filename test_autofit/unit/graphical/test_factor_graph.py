import numpy as np
import pytest

import autofit.mapper.variable
from autofit import graphical as mp


def log_sigmoid(x):
    return -np.log1p(np.exp(-x))


def log_phi(x):
    return -x ** 2 / 2 - 0.5 * np.log(2 * np.pi)


def plus_two(x):
    return x + 2


@pytest.fixture(name="y")
def make_y():
    return autofit.mapper.variable.Variable("y")


@pytest.fixture(name="sigmoid")
def make_sigmoid(x):
    return mp.Factor(log_sigmoid, x=x)


@pytest.fixture(name="phi")
def make_phi(x):
    return mp.Factor(log_phi, x=x)


@pytest.fixture(name="compound")
def make_compound(sigmoid, phi):
    return sigmoid * phi


@pytest.fixture(name="plus")
def make_plus(x):
    return mp.Factor(plus_two, x=x)


@pytest.fixture(name="flat_compound")
def make_flat_compound(plus, y, sigmoid):
    g = plus == y
    phi = mp.Factor(log_phi, x=y)
    return phi * g * sigmoid


class TestFactorGraph:
    def test_names(self, sigmoid, phi, compound):
        assert sigmoid.name == "log_sigmoid"
        assert phi.name == "log_phi"
        assert compound.name == "log_sigmoid.log_phi"

    def test_argument(self, sigmoid, phi, compound):
        x = 5

        assert sigmoid(x=x).log_value == -0.006715348489118068
        assert phi(x=x).log_value == -13.418938533204672
        assert compound(x=x).log_value == -13.42565388169379

    def test_broadcast(self, compound):
        length = 2 ** 10
        array = np.linspace(-5, 5, length)
        result = compound(x=array)
        log_value = result.log_value

        assert isinstance(result.log_value, np.ndarray)
        assert log_value.shape == (length,)

    def test_deterministic_variable_name(self, flat_compound):
        print(flat_compound)
        assert (
            str(flat_compound)
            == "(Factor(log_phi)(x=y) * (Factor(plus_two)(x=x) == (y)) * Factor(log_sigmoid)(x=x))"
        )

    def test_deterministic_variable_value(self, flat_compound, y):
        x = 3
        value = flat_compound(x=x)

        assert value.log_value == -13.467525884778414
        assert value.deterministic_values == {y: 5}

    def test_plates(self):
        obs = autofit.mapper.variable.Plate(name="obs")
        dims = autofit.mapper.variable.Plate(name="dims")

        def sub(a, b):
            return a - b

        x = autofit.mapper.variable.Variable("a", obs, dims)
        y = autofit.mapper.variable.Variable("b", dims)

        subtract = mp.Factor(sub, a=x, b=y)

        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2, 1])

        value = subtract(a=x, b=y).log_value

        assert (value == np.array([[0, 0, 2], [3, 3, 5]])).all()

    @pytest.mark.parametrize("coefficient", [1, 2, 3, 4, 5])
    def test_jacobian(self, x, coefficient):
        factor = mp.Factor(lambda p: coefficient * p, p=x)

        assert factor.jacobian([x], {x: 2}).log_value[x] == pytest.approx(coefficient)
