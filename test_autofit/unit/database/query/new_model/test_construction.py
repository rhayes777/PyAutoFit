from autofit.database import query_model as q
from autofit.mock import mock


def test_simple(aggregator):
    assert aggregator.centre == q.Q("centre")


def test_with_value(aggregator):
    assert (aggregator.centre == 1) == q.Q("centre", q.V("=", 1))


def test_second_level(aggregator):
    assert (aggregator.lens.centre == 1) == q.Q("lens", q.Q("centre", q.V("=", 1)))


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
