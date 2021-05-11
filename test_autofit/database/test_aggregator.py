from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autofit import database as db, NestSamples
from autofit.database.aggregator import Aggregator

directory = Path(__file__).parent


@pytest.fixture(
    name="session",
    scope="module"
)
def make_session():
    engine = create_engine('sqlite://')
    session = sessionmaker(bind=engine)()
    db.Base.metadata.create_all(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(
    name="aggregator",
    scope="module"
)
def make_aggregator(
        session
):
    aggregator = Aggregator(
        session
    )
    aggregator.add_directory(
        str(directory)
    )
    return aggregator


@pytest.fixture(
    name="fit"
)
def make_fit(
        aggregator
):
    return aggregator[0]


def test_name_attributes(fit):
    assert fit.info["dataset"] == "dataset"


def test_samples(
        fit
):
    assert fit["samples"]


def test_load(aggregator):
    assert len(aggregator) == 1
    assert aggregator[0].is_complete is True


def test_values(
        aggregator,
        fit
):
    sample, = aggregator.values(
        "samples"
    )
    assert isinstance(
        sample,
        NestSamples
    )
