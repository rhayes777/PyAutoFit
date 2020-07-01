import pytest
from scipy import stats

import autofit.message_passing.factor_graphs.factor
from autofit import message_passing as mp


@pytest.fixture(
    name="x"
)
def make_x():
    return autofit.message_passing.factor_graphs.factor.Variable("x")


@pytest.fixture(
    name="probit_factor"
)
def make_probit_factor(x):
    return autofit.message_passing.factor_graphs.factor.Factor(
        stats.norm(
            loc=0.,
            scale=1.
        ).logcdf
    )(x)
