import numpy as np
import pytest

from autofit import graphical as mp


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


@pytest.fixture(
    name="model"
)
def make_model(
        prior_a,
        prior_b,
        likelihood_factor,
        linear_factor
):
    return likelihood_factor * linear_factor * prior_a * prior_b


@pytest.fixture(
    name="model_approx"
)
def make_model_approx(
        model,
        a_,
        b_,
        z_,
        x_,
        y_,
):
    a = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])

    n_obs = 200
    n_features, n_dims = a.shape

    x = 2 * np.random.randn(n_obs, n_features)
    z = x.dot(a) + b

    p = 1 / (1 + np.exp(-z))

    y = np.random.binomial(1, p)

    return mp.EPMeanField.from_approx_dists(
        model,
        {
            a_: mp.NormalMessage.from_mode(
                np.zeros((n_features, n_dims)), 10),
            b_: mp.NormalMessage.from_mode(
                np.zeros(n_dims), 10),
            z_: mp.NormalMessage.from_mode(
                np.zeros((n_obs, n_dims)), 100),
            x_: mp.FixedMessage(x),
            y_: mp.FixedMessage(y)
        }
    )


def test_jacobians(
        model_approx
):
    for factor in model_approx.factor_graph.factors:
        factor_approx = model_approx.factor_approximation(factor)
        opt = mp.optimise.OptFactor.from_approx(factor_approx)
        assert opt.numerically_verify_jacobian(
            100, rtol=1e-2, atol=1e-2), factor


def test_laplace(
        model_approx,
        a_,
        b_,
        y_,
        z_,
):
    laplace = mp.LaplaceFactorOptimiser()
    opt = mp.EPOptimiser(
        model_approx.factor_graph,
        default_optimiser=laplace
    )
    model_approx = opt.run(model_approx)

    q_a = model_approx.mean_field[a_]
    q_b = model_approx.mean_field[b_]

    assert q_a.mu[0] == pytest.approx(-1.2, rel=1)
    assert q_a.sigma[0][0] == pytest.approx(0.09, rel=1)

    assert q_b.mu[0] == pytest.approx(-0.5, rel=1)
    assert q_b.sigma[0] == pytest.approx(0.2, rel=2)

    y = model_approx.mean_field[y_].mean
    y_pred = model_approx.mean_field[z_].mean > 0
    (tpr, fpr), (fnr, tnr) = np.dot(
        np.array([y, 1 - y]).reshape(2, -1),
        np.array([y_pred, 1 - y_pred]).reshape(2, -1).T)

    accuracy = (tpr + tnr) / (tpr + fpr + fnr + tnr)
    assert 0.9 > accuracy > 0.7


def test_importance_sampling(
        model,
        model_approx,
        a_,
        b_,
        y_,
        z_,
):
    sampler = mp.ImportanceSampler(n_samples=500)
    history = {}

    for i in range(5):
        for factor in model.factors:
            # We have reduced the entire EP step into a single function
            model_approx, status = mp.sampling.project_model(
                model_approx,
                factor,
                sampler,
                force_sample=True,
                delta=0.8,
            )

            # save and print current approximation
            history[i, factor] = model_approx

    q_z = model_approx.mean_field[z_]
    y = model_approx.mean_field[y_].mean
    y_pred = q_z.mean > 0
    (tpr, fpr), (fnr, tnr) = np.dot(
        np.array([y, 1 - y]).reshape(2, -1),
        np.array([y_pred, 1 - y_pred]).reshape(2, -1).T)

    accuracy = (tpr + tnr) / (tpr + fpr + fnr + tnr)
    assert 0.9 > accuracy > 0.7
