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


def test_unique_tag(
        aggregator,
        gaussian_1,
        gaussian_2
):
    assert aggregator.query(
        aggregator.unique_tag == "one"
    ) == [gaussian_1]
    assert aggregator.query(
        aggregator.unique_tag == "two"
    ) == [gaussian_2]


def test_contains(
        aggregator,
        gaussian_1,
        gaussian_2
):
    assert aggregator.query(
        aggregator.unique_tag.contains(
            "o"
        )
    ) == [
               gaussian_1,
               gaussian_2
           ]
    assert aggregator.query(
        aggregator.unique_tag.contains(
            "ne"
        )
    ) == [
               gaussian_1,
           ]
