from autofit.database import query_model as q


def test_trivial():
    assert q.Q("a") & q.Q("a") == q.Q("a")


def test_second():
    assert q.Q("a") & q.Q("a", q.Q("b")) == q.Q("a", q.Q("b"))


def test_and_commutativity():
    a_and_b = q.And(q.Q("a"), q.Q("b"))
    combined = a_and_b & q.Q("c")

    assert combined == q.And(q.Q("a"), q.Q("b"), q.Q("c"))
    assert len(combined.conditions) == 3


def test_single_argument():
    assert isinstance(
        q.And(q.Q("a")),
        q.Q
    )
