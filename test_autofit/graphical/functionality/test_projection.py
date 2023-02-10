import numpy as np
import pytest
from scipy import integrate, stats

import autofit.messages.normal
from autofit import graphical as graph


@pytest.fixture(name="q_cavity")
def make_q_cavity():
    return autofit.messages.normal.NormalMessage(-0.5, 0.5)


def test_integration(q_cavity, probit_factor):
    x = np.linspace(-3, 3, 2 ** 10)

    probit = stats.norm(loc=0.0, scale=1.0).cdf(x)
    q = stats.norm(loc=q_cavity.mean, scale=q_cavity.sigma).pdf(x)
    tilted_distribution = probit * q

    assert tilted_distribution.shape == (2 ** 10,)

    ni_0, ni_1, ni_2 = (
        integrate.trapz(x ** i * tilted_distribution, x) for i in range(3)
    )

    q_numerical = autofit.messages.normal.NormalMessage.from_sufficient_statistics(
        [ni_1 / ni_0, ni_2 / ni_0]
    )

    assert q_numerical.mean == pytest.approx(-0.253, rel=0.01)
    assert q_numerical.sigma == pytest.approx(0.462, rel=0.01)


def test_laplace_method(probit_factor, q_cavity, x):
    mf = graph.MeanField({x: q_cavity})
    probit_approx = graph.FactorApproximation(
        factor=probit_factor,
        cavity_dist=mf,
        factor_dist=mf,
        model_dist=mf,
    )

    opt = graph.LaplaceOptimiser()
    new_dist, s = opt.optimise_approx(probit_approx)
    q_probit_laplace = new_dist[x]

    assert q_probit_laplace.mean == pytest.approx(-0.258, rel=0.01)
    assert q_probit_laplace.sigma == pytest.approx(0.462, rel=0.01)
