import pytest

import autofit as af
from autofit import database as db
from autofit.mock import mock as m


def test_embedded_query(
        session,
        aggregator
):
    model_1 = db.Fit(
        instance=af.Collection(
            gaussian=m.Gaussian(
                centre=1
            )
        )
    )
    model_2 = db.Fit(
        instance=af.Collection(
            gaussian=m.Gaussian(
                centre=2
            )
        )
    )

    session.add_all([
        model_1,
        model_2
    ])

    result = aggregator.query(
        aggregator.centre == 0
    )

    assert result == []

    result = aggregator.query(
        aggregator.gaussian.centre == 1
    )

    assert result == [model_1]

    result = aggregator.query(
        aggregator.gaussian.centre == 2
    )

    assert result == [model_2]


@pytest.fixture(
    name="gaussian_1"
)
def make_gaussian_1():
    return db.Fit(
        instance=m.Gaussian(
            centre=1
        ),
        dataset_name="dataset 1",
        phase_name="phase"
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
        phase_name="phase"
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


def test_query_dataset(
        gaussian_1,
        gaussian_2,
        aggregator
):
    assert aggregator.query(
        aggregator.dataset_name == "dataset 1"
    ) == [gaussian_1]
    assert aggregator.query(
        aggregator.dataset_name == "dataset 2"
    ) == [gaussian_2]
    assert aggregator.query(
        aggregator.dataset_name.contains(
            "dataset"
        )
    ) == [gaussian_1, gaussian_2]


def test_combine(
        aggregator,
        gaussian_1
):
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.centre == 1)
    ) == [gaussian_1]
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 2") & (aggregator.centre == 1)
    ) == []
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.centre == 2)
    ) == []


def test_combine_attributes(
        aggregator,
        gaussian_1,
        gaussian_2
):
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.phase_name == "phase")
    ) == [gaussian_1]
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 2") & (aggregator.phase_name == "phase")
    ) == [gaussian_2]
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.phase_name == "face")
    ) == []


def test_query(
        aggregator,
        gaussian_1,
        gaussian_2
):
    result = aggregator.query(
        aggregator.centre == 0
    )

    assert result == []

    result = aggregator.query(
        aggregator.centre == 1
    )

    assert result == [gaussian_1]

    result = aggregator.query(
        aggregator.centre == 2
    )

    assert result == [gaussian_2]
