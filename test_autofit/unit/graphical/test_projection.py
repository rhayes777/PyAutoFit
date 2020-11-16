import numpy as np
import pytest
from scipy import integrate

import autofit.graphical.messages.normal
from autofit import graphical as mp


@pytest.fixture(
    name="q_cavity"
)
def make_q_cavity():
    return autofit.graphical.messages.normal.NormalMessage(-0.5, 0.5)


def test_integration(
        q_cavity,
        probit_factor
):
    x = np.linspace(-3, 3, 2 ** 10)

    probit = np.exp(
        probit_factor(
            {mp.Variable('x'): x}
        ).log_value
    )
    q = q_cavity.pdf(x)
    tilted_distribution = probit * q

    assert tilted_distribution.shape == (2 ** 10,)

    ni_0, ni_1, ni_2 = (
        integrate.trapz(x ** i * tilted_distribution, x) for i in range(3))

    q_numerical = autofit.graphical.messages.normal.NormalMessage.from_sufficient_statistics(
        [ni_1 / ni_0, ni_2 / ni_0]
    )

    assert q_numerical.mu == pytest.approx(-0.253, rel=0.01)
    assert q_numerical.sigma == pytest.approx(0.462, rel=0.01)


def test_importance_sampling(
        q_cavity,
        probit_factor
):
    x_samples = q_cavity.sample(200)

    log_weights = probit_factor({mp.Variable('x'): x_samples}).log_value

    q_importance_sampling = q_cavity.project(x_samples, log_weights)

    assert q_importance_sampling.mu == pytest.approx(-0.284, rel=0.5)
    assert q_importance_sampling.sigma == pytest.approx(0.478, rel=0.5)

    mean = np.exp(log_weights).mean()

    assert mean == pytest.approx(0.318, rel=0.1)


def test_laplace_method(probit_factor, q_cavity, x):
    probit_approx = mp.FactorApproximation(
        factor=probit_factor,
        cavity_dist={x: q_cavity},
        factor_dist={},
        model_dist=mp.MeanField({x: q_cavity}))

    opt_probit = mp.OptFactor.from_approx(probit_approx)
    result = opt_probit.maximise({x: 0.})

    q_probit_laplace = autofit.graphical.messages.normal.NormalMessage.from_mode(
        result.mode[x],
        covariance=result.hess_inv[x]
    )

    assert q_probit_laplace.mu == pytest.approx(-0.258, rel=0.01)
    assert q_probit_laplace.sigma == pytest.approx(0.462, rel=0.01)
