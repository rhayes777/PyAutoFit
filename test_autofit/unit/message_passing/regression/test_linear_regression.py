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


@pytest.fixture(
    name="obs"
)
def make_obs():
    return mp.Plate(name='obs')


@pytest.fixture(
    name="features"
)
def make_features():
    return mp.Plate(name='features')


@pytest.fixture(
    name="dims"
)
def make_dims():
    return mp.Plate(name='dims')


@pytest.fixture(
    name="x_"
)
def make_x_(obs, features):
    return mp.Variable('x', obs, features)


@pytest.fixture(
    name="a_"
)
def make_a_(features, dims):
    return mp.Variable('a', features, dims)


@pytest.fixture(
    name="b_"
)
def make_b_(dims):
    return mp.Variable('b', dims)


@pytest.fixture(
    name="z_"
)
def make_z_(obs, dims):
    return mp.Variable('z', obs, dims)


@pytest.fixture(
    name="y_"
)
def make_y_(obs, dims):
    return mp.Variable('y', obs, dims)


@pytest.fixture(
    name="linear_factor"
)
def make_linear_factor(
        x_, a_, b_, z_
):
    return mp.factor(linear)(x_, a_, b_) == z_


@pytest.fixture(
    name="likelihood_factor"
)
def make_likelihood_factor(likelihood, z_, y_):
    return mp.factor(likelihood)(z_, y_)


@pytest.fixture(
    name="prior_a"
)
def make_prior_a(prior, a_):
    return mp.factor(prior)(a_)


@pytest.fixture(
    name="prior_b"
)
def make_prior_b(prior, b_):
    return mp.factor(prior)(b_)


def test(
        prior_a,
        prior_b,
        likelihood_factor,
        linear_factor
):
    a = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])

    n_obs = 100
    n_features, n_dims = a.shape

    x = 5 * np.random.randn(n_obs, n_features)
    y = x.dot(a) + b + np.random.randn(n_obs, n_dims)

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

    assert q_a.mu[0] == pytest.approx(-1.2, rel=1)
    assert q_a.sigma[0][0] == pytest.approx(0.04, rel=1)

    assert q_b.mu[0] == pytest.approx(-0.5, rel=1)
    assert q_b.sigma[0] == pytest.approx(0.2, rel=1)
