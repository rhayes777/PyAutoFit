import pytest

from autofit import database as db


@pytest.fixture(
    name="aggregator"
)
def make_aggregator(
        session
):
    return db.Aggregator(session)
