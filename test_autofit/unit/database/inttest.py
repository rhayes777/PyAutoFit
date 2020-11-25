import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import autofit as af
from autofit.mock import mock as m
from autofit import database as db

from pathlib import Path


@pytest.fixture(name="session")
def make_session():
    engine = create_engine("sqlite:///test.db")
    session = sessionmaker(bind=engine)()
    db.Base.metadata.create_all(engine)
    yield session
    session.close()
    engine.dispose()


def test_commit(session):
    model = af.PriorModel(
        m.Gaussian
    )
    serialized = db.Object(model)
    session.add(serialized)
    session.commit()


def test_read_in_directory(session):
    aggregator = af.Aggregator(
        Path(__file__).parent.parent.parent.parent.parent / "rjlens"
    )
    for item in aggregator:
        session.add(
            db.Object(
                item.model
            )
        )
    session.commit()

