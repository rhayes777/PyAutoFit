import numpy as np

import autofit.mapper.variable
from autofit import graphical as mp


def func(x):
    return 2 * x

def test_str():
    str(mp.NormalMessage(0., 1.))
    str(mp.NormalMessage(
        np.random.rand(10), np.random.rand(10)))
    str(mp.GammaMessage(1., 1.))
    str(mp.GammaMessage(
        np.random.rand(10), np.random.rand(10)))
    str(mp.FixedMessage(0.))
    str(mp.FixedMessage(0., 1.))
    str(mp.FixedMessage(0., np.random.rand(10)))
    str(mp.FixedMessage(np.random.rand(10)))

def test_():
    x = autofit.mapper.variable.Variable("x")
    factor = mp.Factor(
        func,
        x=x
    )

    model = factor * factor

    value = model({x: np.array([1.0])})
    assert value.log_value == np.array([4.0])
