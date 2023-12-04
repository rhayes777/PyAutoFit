import numpy as np
import pytest
from jax import grad, vmap
from jax._src.tree_util import _registry
from jax import numpy as jnp

import autofit as af


def recreate(o):
    flatten_func, unflatten_func = _registry[type(o)]
    children, aux_data = flatten_func(o)
    return unflatten_func(aux_data, children)


@pytest.fixture(name="gaussian")
def make_gaussian():
    return af.Gaussian(centre=1.0, sigma=1.0, normalization=1.0)


def test_gradient(gaussian, monkeypatch):
    monkeypatch.setattr(af.example.model, "np", jnp)
    gradient = grad(gaussian.f)

    assert float(gradient(1.0)) == 0.0

    gaussian.centre = 2.0
    assert float(gradient(1.0)) != 0.0


def classic(gaussian, size=1000):
    return list(map(gaussian.f, np.arange(size)))


def vmapped(gaussian, size=1000):
    f = vmap(gaussian.f)
    return list(f(np.arange(size)))


def test_vmap(gaussian):
    for _ in range(1):
        assert classic(gaussian) == vmapped(gaussian)


def test_gaussian_prior():
    prior = af.GaussianPrior(mean=1.0, sigma=1.0)

    new = recreate(prior)

    assert new.mean == prior.mean
    assert new.sigma == prior.sigma
    assert new.id == prior.id

    assert new.lower_limit == prior.lower_limit
    assert new.upper_limit == prior.upper_limit


@pytest.fixture(name="model")
def _model():
    return af.Model(
        af.Gaussian,
        centre=af.GaussianPrior(mean=1.0, sigma=1.0),
        normalization=af.GaussianPrior(mean=1.0, sigma=1.0, lower_limit=0.0),
        sigma=af.GaussianPrior(mean=1.0, sigma=1.0, lower_limit=0.0),
    )


def test_model(model):
    new = recreate(model)
    assert new.cls == af.Gaussian

    centre = new.centre
    assert centre.mean == model.centre.mean
    assert centre.sigma == model.centre.sigma
    assert centre.id == model.centre.id


def test_instance(model):
    instance = model.instance_from_prior_medians()
    new = recreate(instance)

    assert isinstance(new, af.Gaussian)

    assert new.centre == instance.centre
    assert new.normalization == instance.normalization
    assert new.sigma == instance.sigma


def test_uniform_prior():
    prior = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)

    new = recreate(prior)

    assert new.lower_limit == prior.lower_limit
    assert new.upper_limit == prior.upper_limit
    assert new.id == prior.id


def test_model_instance(model):
    collection = af.Collection(gaussian=model)
    instance = collection.instance_from_prior_medians()
    new = recreate(instance)

    assert isinstance(new, af.ModelInstance)
    assert isinstance(new.gaussian, af.Gaussian)


def test_collection(model):
    collection = af.Collection(gaussian=model)
    new = recreate(collection)

    assert isinstance(new, af.Collection)
    assert isinstance(new.gaussian, af.Model)

    assert new.gaussian.cls == af.Gaussian

    centre = new.gaussian.centre
    assert centre.mean == model.centre.mean
    assert centre.sigma == model.centre.sigma
    assert centre.id == model.centre.id


class KwargClass:
    """
    @DynamicAttrs
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_kwargs():
    model = af.Model(KwargClass, a=1, b=2)
    instance = model.instance_from_prior_medians()

    assert instance.a == 1
    assert instance.b == 2

    new = recreate(instance)

    assert new.a == instance.a
    assert new.b == instance.b
