from autofit.database import query_model as q


def test_trivial():
    assert q.Q("a") & q.Q("a") == q.Q("a")
