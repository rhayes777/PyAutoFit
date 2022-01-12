
import autofit as af
from autofit.database import query as q

def test_simple(aggregator):
    assert aggregator.model.centre.query == q.Q("centre").query


def test_and(aggregator):
    construction = ((aggregator.model.centre == af.Gaussian) & (aggregator.model.centre.x == 0))
    assert construction.query == q.Q(
        "centre",
        q.And(
            q.T(af.Gaussian),
            q.Q(
                "x",
                q.V(
                    "=", 0
                )
            )
        )
    ).query


def test_or(aggregator):
    construction = ((aggregator.model.centre == af.Gaussian) | (aggregator.model.centre.x == 0))
    assert construction.query == q.Q(
        "centre",
        q.Or(
            q.T(af.Gaussian),
            q.Q(
                "x",
                q.V(
                    "=", 0
                )
            )
        )
    ).query


def test_with_value(aggregator):
    assert (aggregator.model.centre == 1).query == q.Q("centre", q.V("=", 1)).query


def test_second_level(aggregator):
    assert (aggregator.model.lens.centre == 1).query == q.Q("lens", q.Q("centre", q.V("=", 1))).query


def test_third_level(aggregator):
    assert (aggregator.model.lens.centre.x == 1).query == q.Q("lens", q.Q("centre", q.Q("x", q.V("=", 1)))).query


def test_with_type(aggregator):
    assert (aggregator.model.centre == af.Gaussian).query == q.Q("centre", q.T(af.Gaussian)).query


def test_with_string(aggregator):
    assert (aggregator.model.centre == "centre").query == q.Q("centre", q.SV("=", "centre")).query


def test_greater_than(aggregator):
    assert (aggregator.model.centre > 1).query == q.Q("centre", q.V(">", 1)).query


def test_less_than(aggregator):
    assert (aggregator.model.centre < 1).query == q.Q("centre", q.V("<", 1)).query


def test_less_than_equals(aggregator):
    assert (aggregator.model.centre <= 1).query == q.Q("centre", q.V("<=", 1)).query


def test_greater_than_equals(aggregator):
    assert (aggregator.model.centre >= 1).query == q.Q("centre", q.V(">=", 1)).query
