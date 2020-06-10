import numpy as np
import pytest
from scipy import stats, integrate

from autofit import message_passing as mp


def test_probit(x):
    phi_factor = mp.factor(
        stats.norm(
            loc=0.,
            scale=1.
        ).logcdf
    )(x)
    q_cavity = mp.NormalMessage(-0.5, 0.5)

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

    phi_numerical = mp.NormalMessage.from_sufficient_statistics(
        [ni_1 / ni_0, ni_2 / ni_0]
    )

    assert phi_numerical.mu == pytest.approx(-0.253, rel=0.01)
    assert phi_numerical.sigma == pytest.approx(0.462, rel=0.01)
