import itertools

import pytest

import autofit as af
import autofit.graphical as g
from autofit.tools.namer import namer


@pytest.fixture(
    autouse=True
)
def reset_namer():
    namer.reset()
    yield
    namer.reset()
    af.ModelObject._ids = itertools.count()


class Analysis(af.Analysis):
    def __init__(self, value):
        self.value = value

    def log_likelihood_function(self, instance):
        return -(instance.one - self.value) ** 2


@pytest.fixture(name="model_factor")
def make_model_factor():
    model = af.Collection(one=af.UniformPrior())

    return g.AnalysisFactor(model, Analysis(0.5))


@pytest.fixture(name="model_factor_2")
def make_model_factor_2():
    model_2 = af.Collection(one=af.UniformPrior())

    return g.AnalysisFactor(model_2, Analysis(0.0))
