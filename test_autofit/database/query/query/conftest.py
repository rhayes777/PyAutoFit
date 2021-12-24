import pytest

import autofit as af

from autofit import database as db


@pytest.fixture(
    name="gaussian_1"
)
def make_gaussian_1():
    return db.Fit(
        id="gaussian_1",
        instance=af.Gaussian(
            centre=1
        ),
        info={"info": 1},
        is_complete=True,
        unique_tag="one"
    )


@pytest.fixture(
    name="gaussian_2"
)
def make_gaussian_2():
    return db.Fit(
        id="gaussian_2",
        instance=af.Gaussian(
            centre=2
        ),
        info={"info": 2},
        is_complete=False,
        unique_tag="two"
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
