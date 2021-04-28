import pytest

from autofit import database as db


@pytest.fixture(
    name="children"
)
def make_children():
    return [
        db.Fit(
            id=f"child_{i}"
        )
        for i in range(10)
    ]


@pytest.fixture(
    name="grid_fit"
)
def make_grid_fit(children):
    return db.Fit(
        id="grid",
        is_grid_search=True,
        children=children
    )


@pytest.fixture(
    autouse=True
)
def add_to_session(
        grid_fit,
        session
):
    session.add(
        grid_fit
    )
    session.flush()


def test_grid_search(
        aggregator,
        grid_fit
):
    result, = aggregator.query(
        aggregator.is_grid_search
    ).fits

    assert result is grid_fit


def test_children(
        aggregator,
        children
):
    assert aggregator.query(
        aggregator.is_grid_search
    ).children().fits == children
