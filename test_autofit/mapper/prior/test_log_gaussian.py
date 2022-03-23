import pytest

import autofit as af


@pytest.fixture(
    name="log_gaussian"
)
def make_log_gaussian():
    return af.LogGaussianPrior(
        mean=1.0,
        sigma=2.0
    )


@pytest.mark.parametrize(
    "unit, value",
    [
        (0.0, 0.0),
        (0.1, 0.2),
        (0.5, 2.7),
        (0.9, 35),
    ]
)
def test_values(
        log_gaussian,
        unit, value
):
    assert log_gaussian.value_for(unit) == pytest.approx(value, rel=0.1)
