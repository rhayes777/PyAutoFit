import pytest

import autofit as af
from autofit import database as db
from autofit.mock import mock as m


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


def test_attribute_query(
        aggregator,
        centre_query
):
    assert aggregator.centre.string == centre_query


def test_embedded_attribute_query(
        aggregator,
        centre_query
):
    string = f"SELECT parent_id FROM object WHERE name = 'lens' AND id IN ({centre_query})"
    assert aggregator.lens.centre.string == string


def test_equality_query(
        aggregator,
        equality_query
):
    assert (aggregator.centre == 1).string == equality_query


def test_string_equality_query(
        aggregator
):
    string = (
        "SELECT parent_id FROM object, string_value "
        "WHERE name = 'centre' AND value = 'one' AND string_value.id = object.id"
    )
    assert (aggregator.centre == "one").string == string


def test_type_equality_query(
        aggregator
):
    string = (
        "SELECT parent_id FROM object "
        "WHERE name = 'centre' AND class_path = 'autofit.mock.mock.Gaussian'"
    )
    assert (aggregator.centre == m.Gaussian).string == string


def test_embedded_equality_query(
        aggregator,
        equality_query
):
    string = f"SELECT parent_id FROM object WHERE name = 'lens' AND id IN ({equality_query})"
    assert (aggregator.lens.centre == 1).string == string


def test_embedded_query(
        session,
        aggregator
):
    model_1 = db.Object.from_object(
        af.Collection(
            gaussian=m.Gaussian(
                centre=1
            )
        )
    )
    model_2 = db.Object.from_object(
        af.Collection(
            gaussian=m.Gaussian(
                centre=2
            )
        )
    )

    session.add_all([
        model_1,
        model_2
    ])

    result = aggregator.filter(
        aggregator.centre == 0
    )

    assert result == []

    result = aggregator.filter(
        aggregator.gaussian.centre == 1
    )

    assert result == [model_1]

    result = aggregator.filter(
        aggregator.gaussian.centre == 2
    )

    assert result == [model_2]


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

    result = aggregator.filter(
        aggregator.centre == 0
    )

    assert result == []

    result = aggregator.filter(
        aggregator.centre == 1
    )

    assert result == [gaussian_1]

    result = aggregator.filter(
        aggregator.centre == 2
    )

    assert result == [gaussian_2]
