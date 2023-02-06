import pytest
from jax import grad, vmap

import autofit as af
import numpy as np


def recreate(o):
    children, aux_data = o._tree_flatten()
    return type(o)._tree_unflatten(aux_data, children)


@pytest.fixture(name="gaussian")
def make_gaussian():
    return af.Gaussian(centre=1.0, sigma=1.0, normalization=1.0)


def test_gradient(gaussian):
    gradient = grad(gaussian.f)

    assert gradient(1.0) == 0.0

    gaussian.centre = 2.0
    assert gradient(1.0) != 0.0


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


def test_model():
    model = af.Model(
        af.Gaussian,
        centre=af.GaussianPrior(mean=1.0, sigma=1.0),
        normalization=af.GaussianPrior(mean=1.0, sigma=1.0, lower_limit=0.0),
        sigma=af.GaussianPrior(mean=1.0, sigma=1.0, lower_limit=0.0),
    )

    new = recreate(model)
    assert new.cls == af.Gaussian

    centre = new.centre
    assert centre.mean == model.centre.mean
    assert centre.sigma == model.centre.sigma
    assert centre.id == model.centre.id
