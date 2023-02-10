import pytest

import autofit as af
from autofit import database as db
from autofit.non_linear.samples import Samples


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
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
    return af.Collection(
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
            serialized_model, db.Model
        )
        assert serialized_model.cls is af.Gaussian

    def test_deserialize(
            self,
            serialized_model
    ):
        assert serialized_model().cls is af.Gaussian


@pytest.fixture(name="collection_with_cache")
def make_collection_with_cache():
    model = af.Collection(gaussian=af.Gaussian)
    model.freeze()
    _ = model.unique_prior_tuples
    return model


class TestFrozenCache:
    def test_model_with_cache(self):
        gaussian = af.Model(af.Gaussian)
        gaussian.freeze()
        _ = gaussian.unique_prior_tuples
        serialized = db.Object.from_object(
            gaussian
        )
        deserialized = serialized()
        assert deserialized._frozen_cache == {}

    def test_collection_with_cache(self, collection_with_cache):
        serialized = db.Object.from_object(
            collection_with_cache
        )
        deserialized = serialized()
        assert deserialized._frozen_cache == {}

    def test_instance_with_cache(self, collection_with_cache):
        instance = collection_with_cache.instance_from_prior_medians()
        serialized = db.Object.from_object(instance)
        deserialized = serialized()
        assert deserialized._frozen_cache == {}

    def test_instance_no_frozen_cache(self, collection_with_cache):
        instance = collection_with_cache.instance_from_prior_medians()
        assert instance._frozen_cache == {}
        assert not hasattr(
            instance.gaussian,
            "_frozen_cache",
        )


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


def test_none_id(model):
    model.id = None
    model = db.Object.from_object(model)()
    assert model.id is not None


class TestCollection:
    def test_serialize(
            self,
            serialized_collection
    ):
        assert isinstance(
            serialized_collection,
            db.Collection
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
            af.Model
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
    model = af.Model(
        af.Gaussian
    )
    serialized = db.Object.from_object(model)
    session.add(serialized)
    session.commit()
