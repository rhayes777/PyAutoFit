import pytest

import autofit as af
from autofit import database as db


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.Gaussian)


def test_instance_from_prior_medians(model):
    db.Object.from_object(model)()
    db.Object.from_object(af.Gaussian())()
    instance = model.instance_from_prior_medians()
    db.Object.from_object(instance)()


def test_object_to_instance(model):
    assert isinstance(
        db.Object.from_object(model)().instance_from_prior_medians(),
        af.Gaussian,
    )


def test_model_with_parameterless_component():
    child = af.Gaussian()
    model = af.Model(
        af.Gaussian,
        centre=child,
    )
    assert ("centre", child) in model.items()

    model = db.Object.from_object(model)()
    instance = model.instance_from_prior_medians()
    assert isinstance(instance.centre, af.Gaussian)


def test_instance_in_collection():
    collection = af.Collection(gaussian=af.Gaussian())
    assert list(collection.items()) == [("gaussian", af.Gaussian())]


def test_samples_summary_model():
    fit = af.db.Fit()
    model = af.Model(af.Gaussian)
    fit["samples_summary"] = af.Samples(model=model, sample_list=[])
    fit.model = model

    assert fit["samples_summary"].model.cls == af.Gaussian
