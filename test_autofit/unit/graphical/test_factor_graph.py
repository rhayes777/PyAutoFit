import numpy as np
import pytest

from scipy.optimize import approx_fprime

import autofit.mapper.variable
from autofit import graphical as mp


def log_sigmoid(x):
    return - np.log1p(np.exp(-x))


def log_phi(x):
    return -x ** 2 / 2 - 0.5 * np.log(2 * np.pi)


def plus_two(x):
    return x + 2

@pytest.fixture(
    name="x"
)
def make_x():
    return mp.Variable('x')

@pytest.fixture(
    name="y"
)
def make_y():
    return mp.Variable('y')


@pytest.fixture(
    name="sigmoid"
)
def make_sigmoid(x):
    return mp.Factor(
        log_sigmoid,
        x=x
    )


@pytest.fixture(
    name="vectorised_sigmoid"
)
def make_vectorised_sigmoid(x):
    return mp.Factor(
        log_sigmoid,
        vectorised=True,
        x=x
    )

@pytest.fixture(
    name="phi"
)
def make_phi(x):
    return mp.Factor(
        log_phi,
        x=x
    )


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
    return mp.Factor(
        plus_two,
        x=x
    )


@pytest.fixture(
    name="flat_compound"
)
def make_flat_compound(
        plus,
        y,
        sigmoid
):
    g = plus == y
    phi = mp.Factor(
        log_phi,
        x=y
    )
    return phi * g * sigmoid


def test_factor_jacobian():
    shape = 4, 3
    z_ = mp.Variable('z', *(mp.Plate() for _ in shape))
    likelihood = mp.NormalMessage(
        np.random.randn(*shape), 
        np.random.exponential(size=shape))
    likelihood_factor = likelihood.as_factor(z_)

    values = {z_: likelihood.sample()}
    fval, jval = likelihood_factor.func_jacobian(
        values, axis=None)
    ngrad = approx_fprime(
        values[z_].ravel(), 
        lambda x: likelihood.logpdf(x.reshape(*shape)).sum(), 
        1e-8).reshape(*shape)
    assert np.allclose(ngrad, jval[z_])


class TestFactorGraph:
    def test_names(
            self,
            sigmoid,
            phi,
            compound
    ):
        assert sigmoid.name == "log_sigmoid"
        assert phi.name == "log_phi"
        assert compound.name == "(log_sigmoid*log_phi)"

    def test_argument(
            self,
            sigmoid,
            phi,
            compound
    ):
        values = {mp.Variable('x') : 5}
        assert sigmoid(values).log_value == -0.006715348489118068
        assert phi(values).log_value == -13.418938533204672
        assert compound(values).log_value == -13.42565388169379
        
    def test_factor_shape(
            self,
            sigmoid,
            phi,
            compound
    ):
        values = {mp.Variable('x') : [5]}
        assert sigmoid(values).log_value[0] == -0.006715348489118068
        assert phi(values).log_value[0] == -13.418938533204672
        assert compound(values).log_value[0] == -13.42565388169379

    def test_multivariate_message(
            self):
        p1, p2, p3 = mp.Plate(), mp.Plate(), mp.Plate()
        x_ = mp.Variable('x', p3, p1)
        y_ = mp.Variable('y', p1, p2)
        z_ = mp.Variable('z', p2, p3)
        
        n1, n2, n3 = shape = (2, 3, 4)

        def sumxyz(x, y, z):
            return (
                np.moveaxis(x[:, :, None], 0, 2) + y[:, :, None] + z[None])

        factor = mp.Factor(sumxyz, x=x_, y=y_, z=z_)

        x = np.arange(n3 * n1).reshape(n3, n1) * 0.1
        y = np.arange(n1 * n2).reshape(n1, n2) * 0.2
        z = np.arange(n2 * n3).reshape(n2, n3) * 0.3
        sumxyz(x, y, z)

        variables = {x_: x, y_: y, z_: z}
        factor(variables)

        model_dist = mp.MeanField({
            x_: mp.NormalMessage(x, 1*np.ones_like(x)),
            y_: mp.NormalMessage(y, 1*np.ones_like(y)),
            z_: mp.NormalMessage(z, 1*np.ones_like(z)),
        })
        
        assert model_dist(variables).log_value.shape == shape
        
    def test_vectorisation(
            self, 
            sigmoid,
            vectorised_sigmoid
    ):
        variables = {mp.Variable('x'): np.full(1000, 5.)}
        assert np.allclose(
            sigmoid(variables).log_value, 
            vectorised_sigmoid(variables).log_value)

    def test_broadcast(
            self,
            compound
    ):
        length = 2 ** 10
        array = np.linspace(-5, 5, length)
        variables = {mp.Variable('x'): array}
        result = compound(variables)
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
        print(flat_compound)
        assert str(
            flat_compound
        ) == "(Factor(log_phi, x=y) * (Factor(plus_two, x=x) == (y)) * Factor(log_sigmoid, x=x))"

    def test_deterministic_variable_value(
            self,
            flat_compound,
            x,
            y
    ):
        value = flat_compound({x: 3})

        assert value.log_value == -13.467525884778414
        assert value.deterministic_values == {
            y: 5
        }

    def test_plates(self):
        obs = autofit.mapper.variable.Plate(name='obs')
        dims = autofit.mapper.variable.Plate(name='dims')

        def sub(a, b):
            return a - b

        a = autofit.mapper.variable.Variable('a', obs, dims)
        b = autofit.mapper.variable.Variable('b', dims)

        subtract = mp.Factor(sub, a=a, b=b)

        x = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        y = np.array([1, 2, 1])

        value = subtract({a: x, b: y}).log_value

        assert (value == x - y).all()

    @pytest.mark.parametrize(
        "coefficient",
        [1, 2, 3, 4, 5]
    )
    def test_jacobian(self, x, coefficient):
        factor = mp.Factor(
            lambda p: coefficient * p,
            p=x
        )

        assert factor.jacobian(
            {x: 2},
            [x],
        )[x] == pytest.approx(coefficient)
