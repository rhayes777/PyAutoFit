import numpy as np
import pytest
from scipy.optimize import approx_fprime

import autofit.mapper.variable
from autofit.messages.normal import NormalMessage
from autofit.mapper.variable import Variable, Plate
from autofit import graphical as graph


def log_sigmoid(x):
    return -np.log1p(np.exp(-x))


def log_phi(x):
    return -(x ** 2) / 2 - 0.5 * np.log(2 * np.pi)


def plus_two(x):
    return x + 2


@pytest.fixture(name="x")
def make_x():
    return Variable("x")


@pytest.fixture(name="y")
def make_y():
    return Variable("y")


@pytest.fixture(name="sigmoid")
def make_sigmoid(x):
    return graph.Factor(log_sigmoid, x=x)


@pytest.fixture(name="vectorised_sigmoid")
def make_vectorised_sigmoid(x):
    return graph.Factor(log_sigmoid, vectorised=True, x=x)


@pytest.fixture(name="phi")
def make_phi(x):
    return graph.Factor(log_phi, x=x)


@pytest.fixture(name="compound")
def make_compound(sigmoid, phi):
    return sigmoid * phi


@pytest.fixture(name="plus")
def make_plus(x):
    return graph.Factor(plus_two, x=x)


@pytest.fixture(name="flat_compound")
def make_flat_compound(plus, y, sigmoid):
    g = plus == y
    phi = graph.Factor(log_phi, x=y)
    return phi * g * sigmoid


def test_factor_jacobian():
    shape = 4, 3
    z_ = Variable("z", *(Plate() for _ in shape))
    likelihood = NormalMessage(
        np.random.randn(*shape), np.random.exponential(size=shape)
    )
    likelihood_factor = likelihood.as_factor(z_)

    values = {z_: likelihood.sample()}
    fval, jval = likelihood_factor.func_jacobian(values)
    ngrad = approx_fprime(
        values[z_].ravel(), lambda x: likelihood.logpdf(x.reshape(*shape)).sum(), 1e-8
    ).reshape(*shape)
    assert np.allclose(ngrad, jval[z_])


class TestFactorGraph:
    def test_names(self, sigmoid, phi, compound):
        assert sigmoid.name == "log_sigmoid"
        assert phi.name == "log_phi"
        assert compound.name == "(log_sigmoid*log_phi)"

    def test_argument(self, sigmoid, phi, compound):
        values = {Variable("x"): 5}
        assert sigmoid(values).log_value == -0.006715348489118068
        assert phi(values).log_value == -13.418938533204672
        assert compound(values).log_value == -13.42565388169379

    def test_factor_shape(self, sigmoid, phi, compound):
        values = {Variable("x"): [5]}
        assert sigmoid(values).log_value == -0.006715348489118068
        assert phi(values).log_value == -13.418938533204672
        assert compound(values).log_value == -13.42565388169379

    def test_vectorisation(self, sigmoid, vectorised_sigmoid):
        variables = {Variable("x"): np.full(1000, 5.0)}
        assert np.allclose(
            sigmoid(variables).log_value, vectorised_sigmoid(variables).log_value
        )

    def test_deterministic_variable_name(self, flat_compound):
        print(flat_compound)
        assert (
            str(flat_compound)
            == "(Factor(log_phi, x=y) * (Factor(plus_two, x=x) == (y)) * Factor(log_sigmoid, x=x))"
        )

    def test_deterministic_variable_value(self, flat_compound, x, y):
        value = flat_compound({x: 3})

        assert value.log_value == -13.467525884778414
        assert value.deterministic_values == {y: 5}

    @pytest.mark.parametrize("coefficient", [1, 2, 3, 4, 5])
    def test_jacobian(self, x, coefficient):
        factor = graph.Factor(lambda p: coefficient * p, p=x)

        assert (
            factor.jacobian(
                {x: 2},
                [x],
            )[x]
            == pytest.approx(coefficient)
        )
