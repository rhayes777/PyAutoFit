import numpy as np
import pytest

from autofit import graphical as mp


@pytest.fixture(name="likelihood")
def make_likelihood(norm):
    def likelihood(z, y):
        return norm.logpdf(z - y)

    return likelihood


@pytest.fixture(name="model")
def make_model(likelihood_factor, linear_factor, prior_a, prior_b):
    return likelihood_factor * linear_factor * prior_a * prior_b


@pytest.fixture(name="model_approx")
def make_model_approx(model, a_, b_, z_, x_, y_):
    a = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])

    n_obs = 100
    n_features, n_dims = a.shape

    x = 5 * np.random.randn(n_obs, n_features)
    y = x.dot(a) + b + np.random.randn(n_obs, n_dims)

    message_a = mp.NormalMessage.from_mode(np.zeros((n_features, n_dims)), 100)

    message_b = mp.NormalMessage.from_mode(np.zeros(n_dims), 100)

    message_z = mp.NormalMessage.from_mode(np.zeros((n_obs, n_dims)), 100)

    return mp.MeanFieldApproximation.from_kws(
        model,
        {
            a_: message_a,
            b_: message_b,
            z_: message_z,
            x_: mp.FixedMessage(x),
            y_: mp.FixedMessage(y),
        },
    )


def test_laplace(model_approx, a_, b_):
    opt = mp.optimise.LaplaceOptimiser(n_iter=3)
    model_approx = opt.run(model_approx)

    q_a = model_approx[a_]
    q_b = model_approx[b_]

    assert q_a.mu[0] == pytest.approx(-1.2, rel=1)
    assert q_a.sigma[0][0] == pytest.approx(0.04, rel=1)

    assert q_b.mu[0] == pytest.approx(-0.5, rel=1)
    assert q_b.sigma[0] == pytest.approx(0.2, rel=1)


def test_importance_sampling(model, model_approx, a_, b_):
    sampler = mp.ImportanceSampler(n_samples=500)

    history = {}
    n_iter = 3

    for i in range(n_iter):
        for factor in model.factors:
            # We have reduced the entire EP step into a single function
            model_approx, status = mp.sampling.project_model(
                model_approx, factor, sampler, force_sample=False, delta=1.0
            )

            # save and print current approximation
            history[i, factor] = model_approx

    q_a = model_approx[a_]
    q_b = model_approx[b_]

    assert q_a.mu[0] == pytest.approx(-1.2, rel=1)
    assert q_a.sigma[0][0] == pytest.approx(7.13, rel=1)

    assert q_b.mu[0] == pytest.approx(-0.5, rel=1)
    assert q_b.sigma[0] == pytest.approx(6.8, rel=1)
