import numpy as np
import pytest
from scipy import stats

import autofit.message_passing.factor_graphs.factor
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
    return autofit.message_passing.factor_graphs.factor.Plate(name='obs')


@pytest.fixture(
    name="features"
)
def make_features():
    return autofit.message_passing.factor_graphs.factor.Plate(name='features')


@pytest.fixture(
    name="dims"
)
def make_dims():
    return autofit.message_passing.factor_graphs.factor.Plate(name='dims')


@pytest.fixture(
    name="x_"
)
def make_x_(obs, features):
    return autofit.message_passing.factor_graphs.factor.Variable('x', obs, features)


@pytest.fixture(
    name="a_"
)
def make_a_(features, dims):
    return autofit.message_passing.factor_graphs.factor.Variable('a', features, dims)


@pytest.fixture(
    name="b_"
)
def make_b_(dims):
    return autofit.message_passing.factor_graphs.factor.Variable('b', dims)


@pytest.fixture(
    name="z_"
)
def make_z_(obs, dims):
    return autofit.message_passing.factor_graphs.factor.Variable('z', obs, dims)


@pytest.fixture(
    name="y_"
)
def make_y_(obs, dims):
    return autofit.message_passing.factor_graphs.factor.Variable('y', obs, dims)


@pytest.fixture(
    name="linear_factor"
)
def make_linear_factor(
        x_, a_, b_, z_
):
    return autofit.message_passing.factor_graphs.factor.Factor(linear)(x_, a_, b_) == z_


@pytest.fixture(
    name="prior_a"
)
def make_prior_a(prior, a_):
    return autofit.message_passing.factor_graphs.factor.Factor(prior)(a_)


@pytest.fixture(
    name="prior_b"
)
def make_prior_b(prior, b_):
    return autofit.message_passing.factor_graphs.factor.Factor(prior)(b_)


@pytest.fixture(
    name="likelihood_factor"
)
def make_likelihood_factor(likelihood, z_, y_):
    return autofit.message_passing.factor_graphs.factor.Factor(likelihood)(z_, y_)