from autofit.database import query_model as q


def test_simple(aggregator):
    assert aggregator.centre == q.Q("centre")


def test_with_value(aggregator):
    assert (aggregator.centre == 1) == q.Q("centre", q.V("=", 1))


def test_second_level(aggregator):
    assert (aggregator.lens.centre == 1) == q.Q("lens", q.Q("centre", q.V("=", 1)))
