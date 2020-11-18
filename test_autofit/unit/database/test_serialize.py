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
    return db.Object(
        model
    )


class TestModel:
    def test_serialise(
            self,
            serialized_model
    ):
        assert isinstance(
            serialized_model, db.PriorModel
        )
        assert serialized_model.cls is m.Gaussian

    def test_deserialize(
            self,
            serialized_model
    ):
        assert serialized_model().cls is m.Gaussian


class TestPriors:
    def test_serialize(
            self,
            serialized_model,
    ):
        assert len(serialized_model.priors) == 3

    def test_deserialize(
            self,
            serialized_model
    ):
        model = serialized_model()
        assert len(model.priors) == 3
        assert isinstance(
            model.centre,
            af.UniformPrior
        )


def test_commit(session):
    model = af.PriorModel(
        m.Gaussian
    )
    serialized = db.Object(model)
    session.add(serialized)
    session.commit()
