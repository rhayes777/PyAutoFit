from autofit.database import query_model as q


def test_named_query():
    query = q.NamedQuery(
        "a"
    )

    assert query == (
        "SELECT parent_id "
        "FROM object as o "
        "WHERE name = 'a'"
    )
