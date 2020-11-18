import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import autofit as af
from autofit.mock import mock as m
from autofit import database as db


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

