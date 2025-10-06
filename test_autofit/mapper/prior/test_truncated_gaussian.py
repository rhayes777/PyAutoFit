import numpy as np
import pytest

import autofit as af


@pytest.fixture(name="truncated_gaussian")
def make_truncated_gaussian():
    return af.TruncatedGaussianPrior(mean=1.0, sigma=2.0, lower_limit=0.95, upper_limit=1.05)


@pytest.mark.parametrize(
    "unit, value",
    [
        (0.001, 0.95),
        (0.5, 1.0),
        (0.999, 1.05),
    ],
)
def test__values(truncated_gaussian, unit, value):

    assert truncated_gaussian.value_for(unit) == pytest.approx(value, rel=0.1)

@pytest.mark.parametrize(
    "unit, value",
    [
        (0.01, -np.inf),
        (1.0, 2.3026892553),
        (2.0, -np.inf),
    ],
)
def test__log_prior_from_value(truncated_gaussian, unit, value):

    assert truncated_gaussian.log_prior_from_value(unit) == pytest.approx(value, rel=0.1)


