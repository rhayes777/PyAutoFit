from operator import sub

import numpy as np
import pytest

import autofit.message_passing.factor_graphs.factor
from autofit import message_passing as mp


def log_sigmoid(x):
    return - np.log1p(np.exp(-x))


def log_phi(x):
    return -x ** 2 / 2 - 0.5 * np.log(2 * np.pi)


def plus_two(x):
    return x + 2


@pytest.fixture(
    name="y"
)
def make_y():
    return autofit.message_passing.factor_graphs.factor.Variable('y')


@pytest.fixture(
    name="sigmoid"
)
def make_sigmoid(x):
    return autofit.message_passing.factor_graphs.factor.Factor(
        log_sigmoid
    )(x)


@pytest.fixture(
    name="phi"
)
def make_phi(x):
    return autofit.message_passing.factor_graphs.factor.Factor(
        log_phi
    )(x)


@pytest.fixture(
    name="compound"
)
def make_compound(
        sigmoid, phi
):
    return sigmoid * phi


@pytest.fixture(
    name="plus"
)
def make_plus(x):
    return autofit.message_passing.factor_graphs.factor.Factor(plus_two)(x)


@pytest.fixture(
    name="flat_compound"
)
def make_flat_compound(
        plus,
        y,
        sigmoid
):
    g = plus == y
    phi = autofit.message_passing.factor_graphs.factor.Factor(log_phi)(y)
    return phi * g * sigmoid


class TestFactorGraph:
    def test_names(
            self,
            sigmoid,
            phi,
            compound
    ):
        assert sigmoid.name == "log_sigmoid"
        assert phi.name == "log_phi"
        assert compound.name == "log_sigmoid.log_phi"

    def test_argument(
            self,
            sigmoid,
            phi,
            compound
    ):
        x = 5

        assert sigmoid(x).log_value == -0.006715348489118068
        assert phi(x).log_value == -13.418938533204672
        assert compound(x).log_value == -13.42565388169379

    def test_broadcast(
            self,
            compound
    ):
        length = 2 ** 10
        array = np.linspace(-5, 5, length)
        result = compound(array)
        log_value = result.log_value

        assert isinstance(
            result.log_value,
            np.ndarray
        )
        assert log_value.shape == (length,)

    def test_deterministic_variable_name(
            self,
            flat_compound
    ):
        assert str(flat_compound) == "(Factor(log_phi)(y) * (Factor(plus_two)(x) == (y)) * Factor(log_sigmoid)(x))"

    def test_deterministic_variable_value(
            self,
            flat_compound
    ):
        x = 3
        value = flat_compound(x=x)

        assert value.log_value == -13.467525884778414
        assert value.deterministic_values == {
            "y": 5
        }

    def test_plates(self):
        obs = autofit.message_passing.factor_graphs.factor.Plate(name='obs')
        dims = autofit.message_passing.factor_graphs.factor.Plate(name='dims')

        x = autofit.message_passing.factor_graphs.factor.Variable('x', obs, dims)
        y = autofit.message_passing.factor_graphs.factor.Variable('y', dims)

        subtract = autofit.message_passing.factor_graphs.factor.Factor(sub)(x, y)

        x = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        y = np.array([1, 2, 1])

        value = subtract(x, y).log_value

        assert (value == np.array(
            [[0, 0, 2],
             [3, 3, 5]]
        )).all()

    @pytest.mark.parametrize(
        "coefficient",
        [1, 2, 3, 4, 5]
    )
    def test_jacobian(self, x, coefficient):
        factor = autofit.message_passing.factor_graphs.factor.Factor(
            lambda p: coefficient * p
        )(x)

        assert factor.jacobian(
            'x',
            x=2
        ).log_value['x'] == pytest.approx(coefficient)
