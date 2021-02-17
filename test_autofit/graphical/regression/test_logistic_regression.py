import numpy as np
import pytest

from autofit import graphical as mp


def likelihood_jacobian(z, y, _variables=None):
    expz = np.exp(-z)
    logp = -np.log1p(expz)
    log1p = -np.log1p(1 / expz)
    loglike = y * logp + (1 - y) * log1p
    if _variables is None:
        return loglike
    else:
        jacs = ()
        for v in _variables:
            if v == 'z':
                jacs += np.expand_dims(
                    y - 1/(1 + expz),
                    tuple(range(np.ndim(loglike)))),
                
            elif v == 'y':
                jacs += np.expand_dims(
                    logp - log1p,
                    tuple(range(np.ndim(loglike)))),
                
        return loglike, jacs

@pytest.fixture(
    name="likelihood_factor_jac"
)
def make_likelihood_factor_jac(z_, y_, obs, dims):
    factor = mp.FactorJacobian(likelihood_jacobian, z=z_, y=y_)
    factor._plates = (obs, dims)
    return factor

@pytest.fixture(
    name="model"
)
def make_model(
        prior_a,
        prior_b,
        likelihood_factor_jac,
        linear_factor_jac
):
    return likelihood_factor_jac * linear_factor_jac * prior_a * prior_b


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
            10, rtol=1e-2, atol=1e-2), factor


def test_laplace(
        model_approx,
        y_,
        z_,
):
    laplace = mp.LaplaceFactorOptimiser(
        default_opt_kws={'jac': True}
    )
    opt = mp.EPOptimiser(
        model_approx.factor_graph,
        default_optimiser=laplace
    )
    new_approx = opt.run(model_approx)

    y = new_approx.mean_field[y_].mean
    z_pred = new_approx(new_approx.mean_field.mean)[z_]
    y_pred = z_pred > 0
    (tpr, fpr), (fnr, tnr) = np.dot(
        np.array([y, 1 - y]).reshape(2, -1),
        np.array([y_pred, 1 - y_pred]).reshape(2, -1).T)

    accuracy = (tpr + tnr) / (tpr + fpr + fnr + tnr)
    assert 0.95 > accuracy > 0.7


def test_importance_sampling(
        model_approx,
        y_,
        z_,
):
    sampler = mp.ImportanceSampler(n_samples=500)

    print_factor = lambda *args: print(args[0])
    print_status = lambda *args: print(args[2])
    callback = mp.expectation_propagation.EPHistory(
        callbacks=(print_factor,print_status))
    opt = mp.EPOptimiser(
        model_approx.factor_graph,
        default_optimiser=sampler,
        callback=callback
    )
    new_approx = opt.run(model_approx, max_steps=5)

    y = new_approx.mean_field[y_].mean
    z_pred = new_approx(new_approx.mean_field.mean)[z_]
    y_pred = z_pred > 0
    (tpr, fpr), (fnr, tnr) = np.dot(
        np.array([y, 1 - y]).reshape(2, -1),
        np.array([y_pred, 1 - y_pred]).reshape(2, -1).T)

    accuracy = (tpr + tnr) / (tpr + fpr + fnr + tnr)
    assert 0.95 > accuracy > 0.7
