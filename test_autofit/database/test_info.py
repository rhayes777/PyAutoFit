import pytest

from autofit import database as db


@pytest.fixture(
    name="fit"
)
def make_fit():
    return db.Fit(
        id="id",
        info={
            "key": "value"
        }
    )


def test_create(
        fit
):
    assert fit.info["key"] == "value"


@pytest.fixture(
    autouse=True
)
def add_to_session(
        session,
        fit
):
    session.add(fit)
    session.commit()


def test_query(
        aggregator,
        fit
):
    assert aggregator.query(
        aggregator.info["key"] == "value"
    ) == [fit]
