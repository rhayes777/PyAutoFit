import numpy as np
import pytest

from autofit.message_passing import factor_graphs as fg


def log_sigmoid(x):
    return - np.log1p(np.exp(-x))


def log_phi(x):
    return -x ** 2 / 2 - 0.5 * np.log(2 * np.pi)


@pytest.fixture(
    name="x"
)
def make_x():
    return fg.Variable("x")


@pytest.fixture(
    name="sigmoid"
)
def make_sigmoid(x):
    return fg.factor(
        log_sigmoid
    )(x)


@pytest.fixture(
    name="phi"
)
def make_phi(x):
    return fg.factor(
        log_phi
    )(x)


@pytest.fixture(
    name="compound"
)
def make_compound(
        sigmoid, phi
):
    return sigmoid * phi


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
