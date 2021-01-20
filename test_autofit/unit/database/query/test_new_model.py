from autofit.database import query_model as q


def test_named_query():
    query = q.NamedQuery(
        "a"
    )

    assert query == (
        "SELECT parent_id "
        "FROM object AS o "
        "WHERE o.name = 'a'"
    )


def test_with_value():
    query = q.NamedQuery(
        "a",
        q.ValueCondition(
            "=",
            1
        )
    )

    assert query == (
        "SELECT parent_id "
        "FROM object AS o, "
        "value AS v "
        "WHERE o.name = 'a' "
        "AND v.value = 1"
    )
