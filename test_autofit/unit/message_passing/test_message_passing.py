import numpy as np
import pytest
from scipy import stats, integrate

from autofit import message_passing as mp


@pytest.fixture(
    name="q_cavity"
)
def make_q_cavity():
    return mp.NormalMessage(-0.5, 0.5)


@pytest.fixture(
    name="phi_factor"
)
def make_phi_factor(x):
    return mp.factor(
        stats.norm(
            loc=0.,
            scale=1.
        ).logcdf
    )(x)


def test_integration(
        q_cavity,
        phi_factor
):
    x = np.linspace(-3, 3, 2 ** 10)

    phi = np.exp(
        phi_factor(
            x
        ).log_value
    )
    q = q_cavity.pdf(x)
    tilted_distribution = phi * q

    assert tilted_distribution.shape == (2 ** 10,)

    ni_0, ni_1, ni_2 = (
        integrate.trapz(x ** i * tilted_distribution, x) for i in range(3))

    q_numerical = mp.NormalMessage.from_sufficient_statistics(
        [ni_1 / ni_0, ni_2 / ni_0]
    )

    assert q_numerical.mu == pytest.approx(-0.253, rel=0.01)
    assert q_numerical.sigma == pytest.approx(0.462, rel=0.01)


def test_importance_sampling(
        q_cavity,
        phi_factor
):
    x_samples = q_cavity.sample(200)

    log_weights = phi_factor(x_samples).log_value

    q_importance_sampling = q_cavity.project(x_samples, log_weights)

    assert q_importance_sampling.mu == pytest.approx(-0.284, rel=0.1)
    assert q_importance_sampling.sigma == pytest.approx(0.478, rel=0.1)

    mean = np.exp(log_weights).mean()

    assert mean == pytest.approx(0.318, rel=0.1)
