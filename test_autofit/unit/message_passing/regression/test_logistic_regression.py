import numpy as np
import pytest

from autofit import message_passing as mp


@pytest.fixture(
    name="likelihood"
)
def make_likelihood():
    def likelihood(z, y):
        expz = np.exp(-z)
        logp = -np.log1p(expz)
        log1p = -np.log1p(1 / expz)
        return y * logp + (1 - y) * log1p

    return likelihood


def test(
        prior_a,
        prior_b,
        likelihood_factor,
        linear_factor
):
    a = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])

    n_obs = 200
    n_features, n_dims = a.shape

    x = 2 * np.random.randn(n_obs, n_features)
    z = x.dot(a) + b

    p = 1 / (1 + np.exp(-z))

    y = np.random.binomial(1, p)

    model = likelihood_factor * linear_factor * prior_a * prior_b

    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        a=mp.NormalMessage.from_mode(
            np.zeros((n_features, n_dims)), 10),
        b=mp.NormalMessage.from_mode(
            np.zeros(n_dims), 10),
        z=mp.NormalMessage.from_mode(
            np.zeros((n_obs, n_dims)), 10),
        x=mp.FixedMessage(x),
        y=mp.FixedMessage(y))

    np.random.seed(1)
    history = {}
    n_iter = 1

    for i in range(n_iter):
        for factor in model.factors:
            # We have reduced the entire EP step into a single function
            model_approx, status = mp.optimise.laplace_factor_approx(
                model_approx, factor, delta=1.)

            # save and print current approximation
            history[i, factor] = model_approx
