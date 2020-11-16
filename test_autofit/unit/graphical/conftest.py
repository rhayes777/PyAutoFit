import pytest
from scipy import stats

import autofit.mapper.variable
from autofit import graphical as mp


@pytest.fixture(name="x")
def make_x():
    return autofit.mapper.variable.Variable("x")


@pytest.fixture(name="probit_factor")
def make_probit_factor(x):
    return mp.Factor(stats.norm(loc=0.0, scale=1.0).logcdf, x=x)
