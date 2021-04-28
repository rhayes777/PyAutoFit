import pytest

from autofit import database as db
from autofit.mock.mock import Gaussian


@pytest.fixture(
    name="children"
)
def make_children():
    return [
        db.Fit(
            id=f"child_{i}",
            instance=Gaussian(
                centre=i
            )
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
        children=children,
        instance=Gaussian(
            centre=1
        )
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


class TestChildren:
    def test_simple(
            self,
            aggregator,
            children
    ):
        assert aggregator.query(
            aggregator.is_grid_search
        ).children().fits == children

    def test_query_after(
            self,
            aggregator
    ):
        results = aggregator.query(
            aggregator.is_grid_search
        ).children().query(
            aggregator.centre <= 5
        ).fits
        assert len(results) == 6

    def test_query_before(
            self,
            aggregator,
            grid_fit,
            session
    ):
        session.add(
            db.Fit(
                id="grid2",
                is_grid_search=True,
                instance=Gaussian(
                    centre=2
                )
            )
        )
        session.flush()

        parent_aggregator = aggregator.query(
            aggregator.is_grid_search & (aggregator.centre == 1)
        )

        result, = parent_aggregator.fits

        assert result is grid_fit

        child_aggregator = parent_aggregator.children()

        results = child_aggregator.fits
        assert len(results) == 10

        results = aggregator.query(
            aggregator.is_grid_search & (aggregator.centre == 2)
        ).children().fits
        assert len(results) == 0
