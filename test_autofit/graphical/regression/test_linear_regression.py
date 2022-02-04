import numpy as np
import pytest

from autofit.graphical import (
    EPMeanField,
    LaplaceOptimiser,
    EPOptimiser,
)
from autofit.messages import FixedMessage, NormalMessage


@pytest.fixture(name="likelihood")
def make_likelihood(norm):
    def likelihood(z, y):
        return norm.logpdf(z - y).sum()

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


def check_model_approx(model_approx, a_, b_, z_, x_, y_):
    mean_field = model_approx.mean_field
    x = model_approx.mean_field[x_].mean
    y = model_approx.mean_field[y_].mean

    X = np.c_[x, np.ones(len(x))]
    XTX = X.T.dot(X) + np.eye(3) / 10.0
    cov = np.linalg.inv(XTX)

    cov_a = cov[:2, :]
    cov_b = cov[2, :]

    # Analytic results
    mean_a = cov_a.dot(X.T.dot(y))
    mean_b = cov_b.dot(X.T.dot(y))
    a_std = cov_a.diagonal()[:, None] ** 0.5
    b_std = cov_b[[-1]] ** 0.5

    assert mean_field[a_].mean == pytest.approx(mean_a, rel=1e-2)
    assert mean_field[b_].mean == pytest.approx(mean_b, rel=1e-2)
    assert mean_field[a_].sigma == pytest.approx(a_std, rel=0.5)
    assert mean_field[b_].sigma == pytest.approx(b_std, rel=0.5)


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

    # like = NormalMessage(y, np.ones_like(y)).as_factor(z_)
    model = likelihood_factor * linear_factor_jac * prior_a * prior_b

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


def test_laplace(
        model_approx,
        a_,
        b_,
        x_,
        y_,
        z_,
):
    laplace = LaplaceOptimiser()
    opt = EPOptimiser(model_approx.factor_graph, default_optimiser=laplace)
    model_approx = opt.run(model_approx)

    check_model_approx(model_approx, a_, b_, z_, x_, y_)


def test_laplace_jac(
        model_jac_approx,
        a_,
        b_,
        x_,
        y_,
        z_,
):
    laplace = LaplaceOptimiser()
    opt = EPOptimiser(model_jac_approx.factor_graph, default_optimiser=laplace)
    model_approx = opt.run(model_jac_approx)

    check_model_approx(model_approx, a_, b_, z_, x_, y_)
