from autofit.database import query_model as q
from autofit.mock import mock


def test_simple(aggregator):
    assert aggregator.centre == q.Q("centre")


def test_and(aggregator):
    construction = ((aggregator.centre == mock.Gaussian) & (aggregator.centre.x == 0))
    assert construction == q.Q(
        "centre",
        q.And(
            q.T(mock.Gaussian),
            q.Q(
                "x",
                q.V(
                    "=", 0
                )
            )
        )
    )


def test_or(aggregator):
    construction = ((aggregator.centre == mock.Gaussian) | (aggregator.centre.x == 0))
    assert construction == q.Q(
        "centre",
        q.Or(
            q.T(mock.Gaussian),
            q.Q(
                "x",
                q.V(
                    "=", 0
                )
            )
        )
    )


def test_with_value(aggregator):
    assert (aggregator.centre == 1) == q.Q("centre", q.V("=", 1))


def test_second_level(aggregator):
    assert (aggregator.lens.centre == 1) == q.Q("lens", q.Q("centre", q.V("=", 1)))


def test_third_level(aggregator):
    assert (aggregator.lens.centre.x == 1) == q.Q("lens", q.Q("centre", q.Q("x", q.V("=", 1))))


def test_with_type(aggregator):
    assert (aggregator.centre == mock.Gaussian) == q.Q("centre", q.T(mock.Gaussian))


def test_with_string(aggregator):
    assert (aggregator.centre == "centre") == q.Q("centre", q.SV("=", "centre"))


def test_greater_than(aggregator):
    assert (aggregator.centre > 1) == q.Q("centre", q.V(">", 1))


def test_less_than(aggregator):
    assert (aggregator.centre < 1) == q.Q("centre", q.V("<", 1))


def test_less_than_equals(aggregator):
    assert (aggregator.centre <= 1) == q.Q("centre", q.V("<=", 1))


def test_greater_than_equals(aggregator):
    assert (aggregator.centre >= 1) == q.Q("centre", q.V(">=", 1))
