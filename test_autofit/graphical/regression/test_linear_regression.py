import numpy as np
import pytest

from autofit import graphical as graph
from autofit.graphical import (
    EPMeanField,
    LaplaceFactorOptimiser,
    FactorJac,
    EPOptimiser,
    utils,
)
from autofit.messages import FixedMessage, NormalMessage


@pytest.fixture(name="likelihood")
def make_likelihood(norm):
    def likelihood(z, y):
        return norm.logpdf(z - y)

    return likelihood


@pytest.fixture(name="model")
def make_model(likelihood_factor, linear_factor, prior_a, prior_b):
    return likelihood_factor * linear_factor * prior_a * prior_b


@pytest.fixture(name="model_approx")
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

    n_obs = 100
    n_features, n_dims = a.shape

    x = 5 * np.random.randn(n_obs, n_features)
    y = x.dot(a) + b + np.random.randn(n_obs, n_dims)

    message_a = NormalMessage.from_mode(np.zeros((n_features, n_dims)), 100)

    message_b = NormalMessage.from_mode(np.zeros(n_dims), 100)

    message_z = NormalMessage.from_mode(np.zeros((n_obs, n_dims)), 100)

    # return MeanFieldApproximation.from_kws(
    return EPMeanField.from_approx_dists(
        model,
        {
            a_: message_a,
            b_: message_b,
            z_: message_z,
            x_: FixedMessage(x),
            y_: FixedMessage(y),
        },
    )


@pytest.fixture(name="model_jac_approx")
def make_model_jac_approx(
    model, a_, b_, z_, x_, y_, likelihood_factor, linear_factor_jac, prior_a, prior_b
):
    a = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])

    n_obs = 100
    n_features, n_dims = a.shape

    x = 5 * np.random.randn(n_obs, n_features)
    y = x.dot(a) + b + np.random.randn(n_obs, n_dims)

    like = NormalMessage(y, np.ones_like(y)).as_factor(z_)
    model = like * linear_factor_jac * prior_a * prior_b

    model_jac_approx = EPMeanField.from_approx_dists(
        model,
        {
            a_: NormalMessage.from_mode(np.zeros((n_features, n_dims)), 100),
            b_: NormalMessage.from_mode(np.zeros(n_dims), 100),
            z_: NormalMessage.from_mode(np.zeros((n_obs, n_dims)), 100),
            x_: FixedMessage(x),
            y_: FixedMessage(y),
        },
    )
    return model_jac_approx


def test_jacobian(
    a_,
    b_,
    x_,
    z_,
    linear_factor,
    linear_factor_jac,
):
    n, m, d = 5, 4, 3
    x = np.random.rand(n, d)
    a = np.random.rand(d, m)
    b = np.random.rand(m)

    values = {x_: x, a_: a, b_: b}

    g0 = {z_: np.random.rand(n, m)}
    fval0, fjac0 = linear_factor.func_jacobian(values)
    fval1, fjac1 = linear_factor_jac.func_jacobian(values)

    fgrad0 = fjac0.grad(g0)
    fgrad1 = fjac1.grad(g0)

    assert fval0 == fval1
    assert (fgrad0 - fgrad1).norm() < 1e-6
    assert (fval0.deterministic_values - fval1.deterministic_values).norm() == 0


def test_laplace_old(model_approx, a_, b_):
    opt = optimise.LaplaceOptimiser(n_iter=3)
    model_approx, status = opt.run(model_approx)
    # assert status.success

    q_a = model_approx.mean_field[a_]
    q_b = model_approx.mean_field[b_]

    assert q_a.mean[0] == pytest.approx(-1.2, rel=1)
    assert q_a.sigma[0][0] == pytest.approx(0.04, rel=1)

    assert q_b.mean[0] == pytest.approx(-0.5, rel=1)
    assert q_b.sigma[0] == pytest.approx(0.2, rel=1)


def test_laplace(
    model_approx,
    a_,
    b_,
    y_,
    z_,
):
    laplace = LaplaceFactorOptimiser()
    opt = EPOptimiser(model_approx.factor_graph, default_optimiser=laplace)
    model_approx = opt.run(model_approx)

    y = model_approx.mean_field[y_].mean
    y_pred = model_approx.mean_field[z_].mean

    assert utils.r2_score(y, y_pred) > 0.95


def test_laplace_jac(
    model_jac_approx,
):
    laplace = LaplaceFactorOptimiser(default_opt_kws={"jac": True})
    opt = EPOptimiser(model_jac_approx.factor_graph, default_optimiser=laplace)
    approx = opt.run(model_jac_approx)

    like = approx.factors[0]
    y = like._factor.mean
    (z_,) = like.variables
    y_pred = approx.mean_field[z_].mean

    assert utils.r2_score(y, y_pred) > 0.95
