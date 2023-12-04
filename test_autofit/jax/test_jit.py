import jax
from jax import numpy as np

import autofit as af
from test_autofit.graphical.gaussian.model import Analysis, Gaussian, make_data
from test_autofit.graphical.gaussian import model as model_module


import pytest


@pytest.fixture(autouse=True)
def monkeypatch_jax_np(monkeypatch):
    monkeypatch.setattr(model_module, "np", np)


@pytest.fixture(autouse=True, name="model")
def make_model():
    return af.Model(Gaussian)


@pytest.fixture(name="analysis")
def make_analysis():
    x = np.arange(100)
    y = make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)
    return Analysis(x, y)


@pytest.fixture(name="instance")
def make_instance():
    return Gaussian()


def test_jit_likelihood(analysis, instance):
    instance = Gaussian()

    jitted = jax.jit(analysis.log_likelihood_function)

    assert jitted(instance) == analysis.log_likelihood_function(instance)
