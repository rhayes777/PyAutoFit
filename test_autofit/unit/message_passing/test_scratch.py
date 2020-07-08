import numpy as np

from autofit import message_passing as mp


def func(x):
    return 2 * x


def test_():
    x = mp.Variable("x")
    factor = mp.Factor(
        func
    )(x)

    model = factor * factor

    value = model(np.array([1.0]))
    assert value.log_value == np.array([4.0])
