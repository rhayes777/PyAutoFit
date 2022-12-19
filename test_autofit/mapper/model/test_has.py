import pytest

import autofit as af


@pytest.fixture(name="model")
def make_prior_model():
    return af.Model(af.Gaussian)


def test_model_has(model):
    assert model.has(af.Prior)


def test_collection_has(model):
    collection = af.Collection(gaussian=model)

    assert collection.has(af.Prior)
