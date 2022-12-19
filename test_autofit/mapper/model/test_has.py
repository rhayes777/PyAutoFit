import pytest

import autofit as af


@pytest.fixture(name="model")
def make_prior_model():
    return af.Model(af.Gaussian)


@pytest.fixture(name="collection")
def make_collection(model):
    return af.Collection(gaussian=model)


def test_model_has(model):
    assert model.has_instance(af.Prior)


def test_collection_has(collection):
    assert collection.has_instance(af.Prior)


def test_collection_has_model(collection):
    assert collection.has_model(af.Gaussian)


def test_collection_of_collection(collection):
    collection = af.Collection(collection=collection)
    assert collection.has_instance(af.Prior)
    assert collection.has_model(af.Gaussian)
