import pytest
from scipy import stats

from autofit import graphical as graph
from autofit.messages.normal import NormalMessage

x = graph.Variable("x")
graph.Factor(stats.norm(loc=-0.5, scale=0.5).logpdf, x)


@pytest.fixture(name="normal_factor")
def make_normal_factor(x):
    return graph.Factor(stats.norm(loc=-0.5, scale=0.5).logpdf, x)


@pytest.fixture(name="model")
def make_model(probit_factor, normal_factor):
    return probit_factor * normal_factor


@pytest.fixture(name="message")
def make_message():
    return NormalMessage(0, 1)


@pytest.fixture(name="model_approx")
def make_model_approx(model, x, message):
    return graph.EPMeanField.from_kws(model, {x: message})


@pytest.fixture(name="probit_approx")
def make_probit_approx(probit_factor, model_approx):
    return model_approx.factor_approximation(probit_factor)


def test_approximations(probit_approx, model_approx, x, message):
    opt = graph.LaplaceOptimiser()
    probit_model_dist, status = opt.optimise_approx(probit_approx)

    # get updated factor approximation
    probit_project, status = probit_approx.project(probit_model_dist, delta=1.0)

    assert probit_project.model_dist[x].mean == pytest.approx(0.506, rel=0.1)
    assert probit_project.model_dist[x].sigma == pytest.approx(0.814, rel=0.1)

    assert probit_project.factor_dist[x].mean == pytest.approx(1.499, rel=0.1)
    assert probit_project.factor_dist[x].sigma == pytest.approx(1.401, rel=0.1)
