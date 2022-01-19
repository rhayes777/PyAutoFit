import pytest

from autofit.database import query as q


def test_trivial():
    assert (q.Q("a") & q.Q("a")).query == q.Q("a").query


def test_second():
    assert (q.Q("a") & q.Q("a", q.Q("b"))).query == (q.Q("a", q.Q("b"))).query


def test_and_commutativity():
    a_and_b = q.And(q.Q("a"), q.Q("b"))
    combined = a_and_b & q.Q("c")

    assert combined == q.And(q.Q("a"), q.Q("b"), q.Q("c"))
    assert len(combined.conditions) == 3


def test_single_argument():
    assert isinstance(
        q.And(q.Q("a")),
        q.NamedQuery
    )


def test_already_compared(
        aggregator
):
    with pytest.raises(
            AssertionError
    ):
        print((aggregator.model.centre == 1) == 1)

    with pytest.raises(
            AttributeError
    ):
        print((aggregator.model.centre == 1).intesity)
