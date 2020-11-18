import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import autofit as af
from autofit import database as db
from autofit.mock import mock as m


@pytest.fixture(name="session")
def make_session():
    engine = create_engine('sqlite://')
    session = sessionmaker(bind=engine)()
    db.Base.metadata.create_all(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(
    name="model"
)
def make_model():
    return af.PriorModel(
        m.Gaussian
    )


@pytest.fixture(
    name="serialized_model"
)
def make_serialized_model(model):
    return db.PriorModel(
        model
    )


def test_serialise_model(
        serialized_model
):
    assert isinstance(
        serialized_model, db.PriorModel
    )
    assert serialized_model.cls is m.Gaussian


def test_deserialize_model(
        serialized_model
):
    assert isinstance(
        serialized_model(),
        m.Gaussian
    )
