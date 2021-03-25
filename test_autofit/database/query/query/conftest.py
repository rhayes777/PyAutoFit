import pytest

from autofit import database as db
from autofit.mock import mock as m


@pytest.fixture(
    name="gaussian_1"
)
def make_gaussian_1():
    return db.Fit(
        instance=m.Gaussian(
            centre=1
        ),
        info={"info": 1},
        is_complete=True
    )


@pytest.fixture(
    name="gaussian_2"
)
def make_gaussian_2():
    return db.Fit(
        instance=m.Gaussian(
            centre=2
        ),
        info={"info": 2},
        is_complete=False
    )


@pytest.fixture(
    autouse=True
)
def add_to_session(
        gaussian_1,
        gaussian_2,
        session
):
    session.add_all([
        gaussian_1,
        gaussian_2
    ])
    session.commit()
