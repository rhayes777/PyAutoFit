import pytest

from autofit import database as db
from autofit.mock import mock as m


@pytest.fixture(
    name="aggregator"
)
def make_aggregator(
        session
):
    return db.Aggregator(session)


def test_attribute_query(
        aggregator
):
    assert aggregator.centre.string == "SELECT parent_id FROM object WHERE name = 'centre'"


def test_equality_query(
        aggregator
):
    string = "SELECT parent_id FROM object, value WHERE name = 'centre' AND value = 1 AND value.id = object.id"
    assert (aggregator.centre == 1).string == string


def test_query(
        session,
        aggregator
):
    gaussian_1 = db.Object.from_object(
        m.Gaussian(
            centre=1
        )
    )
    gaussian_2 = db.Object.from_object(
        m.Gaussian(
            centre=2
        )
    )

    session.add_all([
        gaussian_1,
        gaussian_2
    ])
    session.commit()

    result = aggregator.filter(
        aggregator.centre == 1
    )

    assert result == [gaussian_1]
