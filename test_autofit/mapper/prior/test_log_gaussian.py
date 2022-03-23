import pickle

import pytest

import autofit as af
from autofit.mapper.identifier import Identifier


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


def test_pickle(log_gaussian):
    loaded = pickle.loads(
        pickle.dumps(log_gaussian)
    )
    assert loaded == log_gaussian


def test_attributes(log_gaussian):
    assert log_gaussian.lower_limit == 0
    assert log_gaussian.upper_limit == float("inf")


def test_identifier(log_gaussian):
    Identifier(log_gaussian)
