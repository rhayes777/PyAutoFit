import pytest
from scipy import stats

from autofit import message_passing as mp


@pytest.fixture(
    name="normal_factor"
)
def make_normal_factor(x):
    return mp.factor(
        stats.norm(
            loc=-0.5,
            scale=0.5
        ).logpdf
    )(x)
