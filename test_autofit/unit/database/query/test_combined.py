import pytest

from autofit.mock import mock as m

"""
- Immutable object returning
- Top level vs lower levels
- Same path vs change in path
"""


def test_trivial(
        aggregator,
        type_equality_query
):
    query = (aggregator.centre == m.Gaussian) & (aggregator.centre == m.Gaussian)
    assert query.string == type_equality_query


def test_top_level(
        aggregator,
        type_equality_query
):
    other_query = (
        "SELECT parent_id "
        "FROM object "
        "WHERE class_path = 'autofit.mock.mock.ComplexClass' "
        "AND name = 'other'"
    )
    string = (
        f"SELECT t0.parent_id FROM "
        f"({type_equality_query}) as t0, "
        f"({other_query}) as t1 "
        f"WHERE t0.parent_id = t1.parent_id"
    )
    query = (aggregator.centre == m.Gaussian) & (aggregator.other == m.ComplexClass)
    assert query.string == string


@pytest.fixture(
    name="b_c_string"
)
def make_b_c_string():
    return (
        "SELECT parent_id "
        "FROM object, value "
        "WHERE id IN ("
        "SELECT parent_id "
        "FROM object, value "
        "WHERE name = 'c' "
        "AND value = 2 "
        "AND value.id = object.id"
        ") "
        "AND name = 'b' "
        "AND value = 1 "
        "AND value.id = object.id"
    )


def test_complicated(
        aggregator,
        b_c_string
):
    query = ((aggregator.b == 1) & (aggregator.b.c == 2))

    assert query.string == b_c_string


# def test_second_level_complicated(
#         aggregator,
#         b_c_string
# ):
#     query = ((aggregator.a.b == 1) & (aggregator.a.b.c == 2))
#
#     assert query.string == (
#         "SELECT parent_id "
#         "FROM object "
#         f"WHERE id IN ({b_c_string}) "
#         "AND name = 'a'"
#     )


@pytest.fixture(
    name="second_level_query"
)
def make_second_level_query(
        equality_query
):
    equality_query_2 = (
        "SELECT parent_id "
        "FROM object, value "
        "WHERE name = 'intensity' "
        "AND value = 0 "
        "AND value.id = object.id"
    )

    return (
        "SELECT parent_id "
        "FROM object "
        f"WHERE id IN ({equality_query}) "
        f"AND id IN ({equality_query_2}) "
        "AND name = 'lens'"
    )


def test_second_level(
        aggregator,
        second_level_query
):
    string = ((aggregator.lens.intensity == 0) & (aggregator.lens.centre == 1)).string

    assert string == second_level_query
