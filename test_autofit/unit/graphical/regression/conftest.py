import numpy as np
import pytest
from scipy import stats

import autofit.mapper.variable
from autofit import graphical as mp


@pytest.fixture(
    name="norm"
)
def make_norm():
    return stats.norm(loc=0, scale=1.)


@pytest.fixture(
    autouse=True
)
def set_seed():
    np.random.seed(1)


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


def linear_jacobian(x, a, b, _variables=('x', 'a', 'b')):
    z = np.matmul(x, a) + np.expand_dims(b, -2)
    
    if _variables is None:
        return z
    else:
        (n, m) = z.shape
        d = np.shape(x)[1]
        inds = None
        jacs = ()
        for v in _variables:
            if v == 'x':
                if inds is None:
                    i, j, k = inds = np.indices((n, m, d))
                    
                jac_x = np.zeros((n, m, n, d))
                jac_x[i, j, i, k] = a[k, j]
                jacs += jac_x,

            if v == 'a':
                if inds is None:
                    i, j, k = inds = np.indices((n, m, d))

                jac_a = np.zeros((n, m, d, m))
                jac_a[i, j, k, j] = x[i, k]
                jacs += jac_a,

            if v == 'b':
                jacs += np.repeat(np.eye(m)[None, ...], n, axis=0),

        return z, jacs

@pytest.fixture(
    name="obs"
)
def make_obs():
    return autofit.mapper.variable.Plate(name='obs')


@pytest.fixture(
    name="features"
)
def make_features():
    return autofit.mapper.variable.Plate(name='features')


@pytest.fixture(
    name="dims"
)
def make_dims():
    return autofit.mapper.variable.Plate(name='dims')


@pytest.fixture(
    name="x_"
)
def make_x_(obs, features):
    return autofit.mapper.variable.Variable('x', obs, features)


@pytest.fixture(
    name="a_"
)
def make_a_(features, dims):
    return autofit.mapper.variable.Variable('a', features, dims)


@pytest.fixture(
    name="b_"
)
def make_b_(dims):
    return autofit.mapper.variable.Variable('b', dims)


@pytest.fixture(
    name="z_"
)
def make_z_(obs, dims):
    return autofit.mapper.variable.Variable('z', obs, dims)


@pytest.fixture(
    name="y_"
)
def make_y_(obs, dims):
    return autofit.mapper.variable.Variable('y', obs, dims)


@pytest.fixture(
    name="linear_factor"
)
def make_linear_factor(
        x_, a_, b_, z_
):
    return mp.Factor(linear, x=x_, a=a_, b=b_) == z_

@pytest.fixture(
    name="linear_factor_jac"
)
def make_linear_factor_jac(
        x_, a_, b_, z_
):
    return mp.FactorJacobian(linear_jacobian, x=x_, a=a_, b=b_) == z_

@pytest.fixture(
    name="prior_a"
)
def make_prior_a(prior, a_):
    return mp.Factor(prior, x=a_)


@pytest.fixture(
    name="prior_b"
)
def make_prior_b(prior, b_):
    return mp.Factor(prior, x=b_)


@pytest.fixture(
    name="likelihood_factor"
)
def make_likelihood_factor(likelihood, z_, y_):
    return mp.Factor(likelihood, z=z_, y=y_)
