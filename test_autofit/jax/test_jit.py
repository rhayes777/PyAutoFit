import jax
from jax import numpy as np

import autofit as af
from test_autofit.graphical.gaussian.model import Analysis, Gaussian, make_data
from test_autofit.graphical.gaussian import model


def test_jit(monkeypatch):
    monkeypatch.setattr(model, "np", np)

    x = np.arange(100)
    y = make_data(Gaussian(centre=50.0, normalization=25.0, sigma=10.0), x)
    analysis = Analysis(x, y)

    instance = Gaussian()
    af.Model(Gaussian)

    analysis.log_likelihood_function(instance)

    jax.jit(analysis.log_likelihood_function)(instance)
