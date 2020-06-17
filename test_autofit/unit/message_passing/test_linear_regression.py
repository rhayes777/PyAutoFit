import numpy as np
import pytest
from scipy import stats

from autofit import message_passing as mp


@pytest.fixture(
    name="norm"
)
def make_norm():
    return stats.norm(loc=0, scale=1.)


@pytest.fixture(
    name="prior_norm"
)
def make_prior_norm():
    return stats.norm(loc=0, scale=10.)


@pytest.fixture(
    name="likelihood"
)
def make_likelihood(norm):
    def likelihood(z, y):
        return norm.logpdf(z - y)

    return likelihood


@pytest.fixture(
    name="prior"
)
def make_prior(prior_norm):
    def prior(x):
        return prior_norm.logpdf(x)

    return prior


def linear(x, a, b):
    return np.matmul(x, a) + np.expand_dims(b, -2)


def test_(
        prior,
        likelihood
):
    a = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])

    n_obs = 100
    n_features, n_dims = a.shape

    x = 5 * np.random.randn(n_obs, n_features)
    y = x.dot(a) + b + np.random.randn(n_obs, n_dims)

    obs = mp.Plate(name='obs')
    features = mp.Plate(name='features')
    dims = mp.Plate(name='dims')

    x_ = mp.Variable('x', obs, features)
    a_ = mp.Variable('a', features, dims)
    b_ = mp.Variable('b', dims)
    z_ = mp.Variable('z', obs, dims)
    y_ = mp.Variable('y', obs, dims)

    # defining deterministic factor
    linear_factor = mp.factor(linear)(x_, a_, b_) == z_
    # likelihood
    likelihood_factor = mp.factor(likelihood)(z_, y_)
    # At the moment we have to define priors for A and b
    prior_a = mp.factor(prior)(a_)
    prior_b = mp.factor(prior)(b_)

    model = likelihood_factor * linear_factor * prior_a * prior_b

    message_a = mp.FracMessage(
        mp.NormalMessage.from_mode(
            np.zeros((n_features, n_dims)),
            100
        )
    )
    message_b = mp.FracMessage(
        mp.NormalMessage.from_mode(
            np.zeros(n_dims),
            100
        )
    )
    message_z = mp.FracMessage(
        mp.NormalMessage.from_mode(
            np.zeros((n_obs, n_dims)),
            100
        )
    )

    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        a=message_a,
        b=message_b,
        z=message_z,
        x=mp.FixedMessage(x),
        y=mp.FixedMessage(y))

    np.random.seed(1)
    history = {}
    n_iter = 3

    for i in range(n_iter):
        for factor in model.factors:
            # We have reduced the entire EP step into a single function
            model_approx, status = mp.optimise.laplace_factor_approx(
                model_approx,
                factor,
                delta=1.
            )

            # save and print current approximation
            history[i, factor] = model_approx

    q_a = model_approx['a'].value.parameters.round(3)
    q_b = model_approx['b'].value.parameters.round(3)

    assert q_a.mu[0] == pytest.approx(-1.243, rel=0.1)
    assert q_a.sigma[0] == pytest.approx(0.021, rel=0.1)

    assert q_b.mu[0] == pytest.approx(-0.507, rel=0.1)
    assert q_b.sigma[0] == pytest.approx(0.102, rel=0.1)
