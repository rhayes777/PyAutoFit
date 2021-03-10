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
        dataset_name="dataset 1",
        phase_name="phase",
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
        dataset_name="dataset 2",
        phase_name="phase",
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
