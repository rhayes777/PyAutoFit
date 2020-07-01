import numpy as np
from scipy import integrate, stats

import autofit.message_passing.factor_graphs.factor
from autofit import message_passing as mp


def test():
    ## define parameters of model
    alpha, beta, gamma, delta = 2 / 3, 4 / 3, 1, 1
    r = np.array([alpha, - gamma])
    A = np.array([
        [0., beta / alpha],
        [delta / gamma, 0.]])
    K = 1
    noise = 0.1

    # starting composition
    y0 = np.array([1., 1.])

    n_species = len(y0)
    n_obs = 30
    t_space = 1.
    t_obs = np.r_[
        0,
        (np.arange(n_obs - 1) * t_space +
         np.random.rand(n_obs - 1)) * t_space]

    def lotka_volterra(t, z, r=r, A=A, K=K):
        return z * r * (1 - A.dot(z) / K)

    def calc_lotka_volterra(y0, r, A, K, t_obs):
        res = integrate.solve_ivp(
            lotka_volterra,
            (t_obs[0], t_obs[-1]), y0, t_eval=t_obs,
            args=(r, A, K),
            method='BDF')

        y_ = res.y
        n = y_.shape[1]
        n_obs = len(t_obs)
        # make sure output is correct dimension
        if n != n_obs:
            y_ = np.c_[
                     y_, np.repeat(y_[:, [-1]],
                                   n_obs - n, axis=1)][:, :n_obs]
            if y_.shape[1] != n_obs:
                raise Exception

        return y_

    y_true = calc_lotka_volterra(y0, r, A, K, t_obs)
    y = y_true + noise * np.random.randn(n_species, n_obs)

    ## Specifying dimensions of problem
    obs = autofit.message_passing.factor_graphs.factor.Plate(name='obs')
    species = autofit.message_passing.factor_graphs.factor.Plate(name='species')
    # Need to specify a second plate for species because
    # A is (species, species) and we need a second plate
    # to unique specify the second dimension
    speciesA = autofit.message_passing.factor_graphs.factor.Plate(name='species')
    dims = autofit.message_passing.factor_graphs.factor.Plate(name='dims')

    ## Specifying variables
    r_ = autofit.message_passing.factor_graphs.factor.Variable('r', species)
    A_ = autofit.message_passing.factor_graphs.factor.Variable('A', species, speciesA)
    K_ = autofit.message_passing.factor_graphs.factor.Variable('K')

    y0_ = autofit.message_passing.factor_graphs.factor.Variable('y0', species)
    y_ = autofit.message_passing.factor_graphs.factor.Variable('y', species, obs)

    y_obs_ = autofit.message_passing.factor_graphs.factor.Variable('y_obs', species, obs)
    t_obs_ = autofit.message_passing.factor_graphs.factor.Variable('t_obs', obs)

    _norm = stats.norm(loc=0, scale=noise)
    _prior = stats.norm(loc=0, scale=10)
    _prior_exp = stats.expon(loc=0, scale=1)

    def _likelihood(y_obs, y):
        return _norm.logpdf(y_obs - y)

    ## Specifying factors

    likelihood = autofit.message_passing.factor_graphs.factor.Factor(_likelihood)(y_obs_, y_)
    prior_A = autofit.message_passing.factor_graphs.factor.Factor(_prior.logpdf, 'prior_A')(A_)
    prior_r = autofit.message_passing.factor_graphs.factor.Factor(_prior.logpdf, 'prior_r')(r_)
    prior_y0 = autofit.message_passing.factor_graphs.factor.Factor(_prior_exp.logpdf, 'prior_y0')(y0_)

    # calc_lotka_volterra does not vectorise over
    # multiple inputs, see `FactorNode._py_vec_call`
    LV = autofit.message_passing.factor_graphs.factor.Factor(
        calc_lotka_volterra, 'LV',
        vectorised=False
    )(y0_, r_, A_, K_, t_obs_) == y_

    ## Defining model
    priors = prior_A * prior_r * prior_y0
    LV_model = (likelihood * LV) * priors
    LV_model._name = 'LV_model'

    model_approx = mp.MeanFieldApproximation.from_kws(
        LV_model,
        A=mp.NormalMessage.from_mode(A, 100.),
        r=mp.NormalMessage.from_mode(r, 100.),
        y0=mp.FracMessage(
            mp.GammaMessage.from_mode(np.ones_like(y0), 1)),
        y=mp.NormalMessage.from_mode(y, 1),
        K=mp.FixedMessage(1),
        y_obs=mp.FixedMessage(y),
        t_obs=mp.FixedMessage(t_obs), )

    history = {}
    n_iter = 1

    factors = [f for f in LV_model.factors if f not in (LV,)]

    np.random.seed(1)

    for i in range(n_iter):
        # perform least squares fit for LV model
        model_approx, status = mp.lstsq_laplace_factor_approx(
            model_approx,
            LV
        )

        # perform laplace non linear fit for other factors
        for factor in factors:
            model_approx, status = mp.optimise.laplace_factor_approx(
                model_approx, factor, delta=1.)
            history[i, factor] = model_approx

    model_mean = {v: d.mean for v, d in model_approx.approx.items()}
    y_pred = LV_model(**model_mean).deterministic_values['y']
    assert y_pred.min() > 0.2
    assert y_pred.max() < 1.6
