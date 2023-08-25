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
