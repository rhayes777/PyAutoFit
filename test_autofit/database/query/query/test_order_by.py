import pytest

import autofit as af
from autofit import database as db


@pytest.fixture(
    name="gaussian_0"
)
def make_gaussian_0(
        session
):
    gaussian_0 = db.Fit(
        id="gaussian_0",
        instance=af.Gaussian(
            centre=1
        ),
        info={"info": 1},
        is_complete=True,
        unique_tag="zero"
    )
    session.add(gaussian_0)
    session.commit()
    return gaussian_0


def test_order_by(
        aggregator,
        gaussian_1,
        gaussian_2,
        gaussian_0
):
    assert aggregator.order_by(
        aggregator.search.unique_tag
    ) == [gaussian_1, gaussian_2, gaussian_0]


def test_reversed(
        aggregator,
        gaussian_1,
        gaussian_2,
        gaussian_0
):
    assert aggregator.order_by(
        aggregator.search.unique_tag,
        reverse=True
    ) == [gaussian_0, gaussian_2, gaussian_1]


def test_boolean(
        aggregator,
        gaussian_1,
        gaussian_2
):
    assert aggregator.order_by(
        aggregator.search.is_complete
    ) == [gaussian_2, gaussian_1]
    assert aggregator.order_by(
        aggregator.search.is_complete,
        reverse=True
    ) == [gaussian_1, gaussian_2]


def test_combined(
        aggregator,
        gaussian_1,
        gaussian_2,
        gaussian_0
):
    assert aggregator.order_by(
        aggregator.search.is_complete
    ).order_by(
        aggregator.search.unique_tag
    ) == [gaussian_2, gaussian_1, gaussian_0]
