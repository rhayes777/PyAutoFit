import numpy as np
from scipy import stats

from autofit import message_passing as mp


def test():
    A = np.array([[-1.3], [0.7]])
    b = np.array([-0.5])

    n_obs = 200
    n_features, n_dims = A.shape

    X = 2 * np.random.randn(n_obs, n_features)
    Z = X.dot(A) + b  # + np.random.randn(n_obs, n_dims)

    p = 1 / (1 + np.exp(-Z))

    y = np.random.binomial(1, p)

    obs = mp.Plate(name='obs')
    features = mp.Plate(name='features')
    dims = mp.Plate(name='dims')

    X_ = mp.Variable('X', obs, features)
    A_ = mp.Variable('A', features, dims)
    b_ = mp.Variable('b', dims)
    Z_ = mp.Variable('Z', obs, dims)
    y_ = mp.Variable('y', obs, dims)

    def _linear(X, A, b):
        return (np.matmul(X, A) + np.expand_dims(b, -2))

    def _likelihood(Z, y):
        expZ = np.exp(-Z)
        logp = -np.log1p(expZ)
        log1p = -np.log1p(1 / expZ)
        return y * logp + (1 - y) * log1p

    _prior_norm = stats.norm(loc=0, scale=100.)

    def _prior(x):
        return _prior_norm.logpdf(x)

    # defining deterministic factor
    linear = mp.factor(_linear)(X_, A_, b_) == Z_
    # likelihood
    likelihood = mp.factor(_likelihood)(Z_, y_)
    # At the moment we have to define priors for A and b
    prior_A = mp.factor(_prior)(A_)
    prior_b = mp.factor(_prior)(b_)

    model = likelihood * linear * prior_A * prior_b
    print(model)
    print(model.call_signature)

    model_approx = mp.MeanFieldApproximation.from_kws(
        model,
        A=mp.NormalMessage.from_mode(
            np.zeros((n_features, n_dims)), 10),
        b=mp.NormalMessage.from_mode(
            np.zeros((n_dims)), 10),
        Z=mp.NormalMessage.from_mode(
            np.zeros((n_obs, n_dims)), 10),
        #     A = mp.FracMessage(mp.NormalMessageBelief.from_mode(
        #         np.zeros((n_features, n_dims)), 100)),
        #     b = mp.FracMessage(mp.NormalMessageBelief.from_mode(
        #         np.zeros((n_dims)), 100)),
        #     Z = mp.FracMessage(mp.NormalMessageBelief.from_mode(
        #         np.zeros((n_obs, n_dims)),
        #         100)),
        X=mp.FixedMessage(X),
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
