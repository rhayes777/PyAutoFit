from autofit import database as db


def test_is_complete(
        gaussian_1,
        aggregator
):
    assert aggregator.query(
        aggregator.is_complete
    ) == [gaussian_1]


def test_is_not_complete(
        gaussian_2,
        aggregator
):
    assert aggregator.query(
        ~aggregator.is_complete
    ) == [gaussian_2]


def test_call(
        gaussian_1,
        aggregator
):
    assert aggregator(
        aggregator.is_complete
    ) == [gaussian_1]


def test_completed_only(
        gaussian_1,
        session
):
    aggregator = db.Aggregator.from_database(
        '',
        completed_only=True
    )
    aggregator.session = session
    assert aggregator == [gaussian_1]


def test_combine(
        aggregator,
        gaussian_1
):
    assert aggregator.query(
        aggregator.is_complete & (aggregator.centre == 1)
    ) == [gaussian_1]
    assert aggregator.query(
        (~aggregator.is_complete) & (aggregator.centre == 1)
    ) == []
    assert aggregator.query(
        aggregator.is_complete & (aggregator.centre == 2)
    ) == []
