import pytest
import numpy as np

import autofit as af
from autofit import database as db
from autofit.database import Fit


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian)


def test_instance_from_prior_medians(model):
    db.Object.from_object(model)()
    db.Object.from_object(af.ex.Gaussian())()
    instance = model.instance_from_prior_medians()
    db.Object.from_object(instance)()


def test_object_to_instance(model):
    assert isinstance(
        db.Object.from_object(model)().instance_from_prior_medians(),
        af.ex.Gaussian,
    )


def test_model_with_parameterless_component():
    child = af.ex.Gaussian()
    model = af.Model(
        af.ex.Gaussian,
        centre=child,
    )
    assert ("centre", child) in model.items()

    model = db.Object.from_object(model)()
    instance = model.instance_from_prior_medians()
    assert isinstance(instance.centre, af.ex.Gaussian)


def test_instance_in_collection():
    collection = af.Collection(gaussian=af.ex.Gaussian())
    assert list(collection.items()) == [("gaussian", af.ex.Gaussian())]


def test_samples_summary_model():
    fit = af.db.Fit()
    model = af.Model(af.ex.Gaussian)
    fit["samples_summary"] = af.Samples(model=model, sample_list=[])
    fit.model = model

    assert fit["samples_summary"].model.cls == af.ex.Gaussian


def test_dict_with_tuple_keys():
    d = {("a", "b"): 1}
    assert db.Object.from_object(d)() == d


def test_persist_values(session):
    fit = Fit(id=1)

    fit.set_pickle("pickle", "test")
    fit.set_array("array", np.array([1, 2, 3]))

    session.add(fit)
    session.commit()

    fit = session.query(Fit).first()

    assert fit["pickle"] == "test"
    assert fit["array"].tolist() == [1, 2, 3]
