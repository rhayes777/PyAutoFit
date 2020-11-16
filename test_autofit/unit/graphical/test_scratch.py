import numpy as np

import autofit.mapper.variable
from autofit import graphical as mp


def func(x):
    return 2 * x


def test_():
    x = autofit.mapper.variable.Variable("x")
    factor = mp.Factor(func, x=x)

    model = factor * factor

    value = model(x=np.array([1.0]))
    assert value.log_value == np.array([4.0])
