import numpy as np
from scipy import stats

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
