import pytest

from autofit import database as db


@pytest.fixture(
    name="aggregator"
)
def make_aggregator(
        session
):
    return db.Aggregator(session)


@pytest.fixture(
    name="centre_query"
)
def make_centre_query():
    return "SELECT parent_id FROM object WHERE name = 'centre'"


@pytest.fixture(
    name="equality_query"
)
def make_equality_query():
    return "SELECT parent_id FROM object, value WHERE name = 'centre' AND value = 1 AND value.id = object.id"


@pytest.fixture(
    name="type_equality_query"
)
def make_type_equality_query():
    return (
        "SELECT parent_id FROM object WHERE class_path = 'autofit.mock.mock.Gaussian' "
        "AND name = 'centre'"
    )
