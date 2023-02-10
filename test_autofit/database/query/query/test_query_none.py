import pytest

import autofit as af
from autofit import database as db


@pytest.fixture(
    name="gaussian_none"
)
def make_gaussian_none(
        session
):
    gaussian_none = db.Fit(
        id="gaussian_none",
        instance=af.Gaussian(
            centre=None
        )
    )
    session.add(gaussian_none)
    session.commit()
    return gaussian_none


def test_query_none(
        gaussian_none,
        aggregator
):
    assert aggregator.query(
        aggregator.model.centre == None
    ) == [gaussian_none]


def test_not(
        gaussian_1,
        gaussian_2,
        aggregator
):
    assert aggregator.query(
        aggregator.model.centre != None
    ) == [gaussian_1, gaussian_2]


def test_combined(
        gaussian_none,
        gaussian_1,
        aggregator
):
    assert aggregator.query(
        (aggregator.model.centre == None) | (aggregator.model.centre == 1.0)
    ) == [gaussian_1, gaussian_none]
