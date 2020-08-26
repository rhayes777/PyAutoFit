import pytest
from scipy import stats


from autofit import expectation_propagation as mp


@pytest.fixture(
    name="x"
)
def make_x():
    return mp.Variable("x")


@pytest.fixture(
    name="probit_factor"
)
def make_probit_factor(x):
    return mp.Factor(
        stats.norm(
            loc=0.,
            scale=1.
        ).logcdf,
        x=x
    )
