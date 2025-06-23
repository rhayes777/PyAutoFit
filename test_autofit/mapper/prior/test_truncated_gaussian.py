import pickle

import pytest

import autofit as af
from autofit.mapper.identifier import Identifier


@pytest.fixture(name="truncated_gaussian")
def make_truncated_gaussian():
    return af.TruncatedGaussianPrior(mean=1.0, sigma=2.0, lower_limit=0.95, upper_limit=1.05)


@pytest.mark.parametrize(
    "unit, value",
    [
        # (0.0, 0.0),
        (0.001, 0.95),
        (0.5, 1.0),
        (0.999, 1.05),
    ],
)
def test_values(truncated_gaussian, unit, value):
    print(unit, value)
    assert truncated_gaussian.value_for(unit) == pytest.approx(value, rel=0.1)

