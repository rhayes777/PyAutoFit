from autofit.mock import mock as m


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


def test_second_level(
        aggregator,
        equality_query
):
    equality_query_2 = (
        "SELECT parent_id "
        "FROM object, value "
        "WHERE name = 'intensity' "
        "AND value = 0 "
        "AND value.id = object.id"
    )

    string = ((aggregator.lens.intensity == 0) & (aggregator.lens.centre == 1)).string

    assert string == (
        f"SELECT parent_id FROM object WHERE name = 'lens' AND id IN ({equality_query_2} AND id in {equality_query})"
    )
