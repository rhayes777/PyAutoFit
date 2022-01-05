import pytest

import autofit as af
from autofit import database as db
from autofit.non_linear.samples import Samples


@pytest.fixture(
    name="model"
)
def make_model():
    return af.PriorModel(
        af.Gaussian
    )


@pytest.fixture(
    name="serialized_model"
)
def make_serialized_model(model):
    return db.Object.from_object(
        model
    )


@pytest.fixture(
    name="collection"
)
def make_collection(model):
    return af.CollectionPriorModel(
        model=model
    )


@pytest.fixture(
    name="serialized_collection"
)
def make_serialized_collection(collection):
    return db.Object.from_object(
        collection
    )


class TestInstance:
    def test_serialize(self):
        serialized_instance = db.Object.from_object(
            af.Gaussian()
        )
        assert len(
            serialized_instance.children
        ) == 3

    def test_tuple(self):
        assert db.Object.from_object(
            (1, 2)
        )() == (1, 2)


class TestModel:
    def test_serialize(
            self,
            serialized_model
    ):
        assert isinstance(
            serialized_model, db.PriorModel
        )
        assert serialized_model.cls is af.Gaussian

    def test_deserialize(
            self,
            serialized_model
    ):
        assert serialized_model().cls is af.Gaussian


class TestPriors:
    def test_serialize(
            self,
            serialized_model,
    ):
        assert len(serialized_model.children) == 3

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


class TestCollection:
    def test_serialize(
            self,
            serialized_collection
    ):
        assert isinstance(
            serialized_collection,
            db.CollectionPriorModel
        )
        child, = serialized_collection.children
        assert len(child.children) == 3

    def test_deserialize(
            self,
            serialized_collection
    ):
        collection = serialized_collection()
        assert len(collection) == 1
        assert isinstance(
            collection.model,
            af.PriorModel
        )


class TestClasses:
    def test_samples(self):
        db.Object.from_object(
            Samples(
                af.ModelMapper(),
                None
            )
        )()

    def test_string(self):
        assert "string" == db.Object.from_object(
            "string"
        )()


class Serialisable:
    def __init__(self):
        self.value = 1

    def __getstate__(self):
        return {
            "value": 2 * self.value
        }

    def __setstate__(self, state):
        state["value"] *= -1
        self.__dict__.update(
            state
        )


def test_get_set_state():
    assert db.Object.from_object(
        Serialisable()
    )().value == -2


def test_none():
    assert db.Object.from_object(None)() is None


def test_commit(session):
    model = af.PriorModel(
        af.Gaussian
    )
    serialized = db.Object.from_object(model)
    session.add(serialized)
    session.commit()
