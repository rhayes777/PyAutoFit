import numpy as np
from scipy import integrate, stats

import autofit.expectation_propagation.messages.fixed
import autofit.expectation_propagation.messages.gamma
import autofit.expectation_propagation.messages.normal
from autofit import expectation_propagation as mp


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
    obs = mp.Plate(name='obs')
    species = mp.Plate(name='species')
    # Need to specify a second plate for species because
    # A is (species, species) and we need a second plate
    # to unique specify the second dimension
    speciesA = mp.Plate(name='species')
    dims = mp.Plate(name='dims')

    ## Specifying variables
    r_ = mp.Variable('r', species)
    A_ = mp.Variable('A', species, speciesA)
    K_ = mp.Variable('K')

    y0_ = mp.Variable('y0', species)
    y_ = mp.Variable('y', species, obs)

    y_obs_ = mp.Variable('y_obs', species, obs)
    t_obs_ = mp.Variable('t_obs', obs)

    _norm = stats.norm(loc=0, scale=noise)
    _prior = stats.norm(loc=0, scale=10)
    _prior_exp = stats.expon(loc=0, scale=1)

    def _likelihood(y_obs, y):
        return _norm.logpdf(y_obs - y)

    ## Specifying factors

    likelihood = mp.Factor(_likelihood, y_obs=y_obs_, y=y_)
    prior_A = mp.Factor(_prior.logpdf, 'prior_A', x=A_)
    prior_r = mp.Factor(_prior.logpdf, 'prior_r', x=r_)
    prior_y0 = mp.Factor(_prior_exp.logpdf, 'prior_y0', x=y0_)

    # calc_lotka_volterra does not vectorise over
    # multiple inputs, see `FactorNode._py_vec_call`
    LV = mp.Factor(
        calc_lotka_volterra, 'LV',
        vectorised=False,
        y0=y0_,
        r=r_,
        A=A_,
        K=K_,
        t_obs=t_obs_
    ) == y_

    ## Defining model
    priors = prior_A * prior_r * prior_y0
    LV_model = (likelihood * LV) * priors
    LV_model._name = 'LV_model'

    model_approx = mp.MeanFieldApproximation.from_kws(
        LV_model,
        {
            A_: autofit.expectation_propagation.messages.normal.NormalMessage.from_mode(A, 100.),
            r_: autofit.expectation_propagation.messages.normal.NormalMessage.from_mode(r, 100.),
            y0_: autofit.expectation_propagation.messages.gamma.GammaMessage.from_mode(np.ones_like(y0), 1),
            y_: autofit.expectation_propagation.messages.normal.NormalMessage.from_mode(y, 1),
            K_: autofit.expectation_propagation.messages.fixed.FixedMessage(1),
            y_obs_: autofit.expectation_propagation.messages.fixed.FixedMessage(y),
            t_obs_: autofit.expectation_propagation.messages.fixed.FixedMessage(t_obs)
        },
    )

    history = {}
    n_iter = 1

    factors = [f for f in LV_model.factors if f not in (LV,)]

    np.random.seed(1)

    opt = mp.optimise.LaplaceOptimiser(
        model_approx,
        n_iter=n_iter
    )

    for i in range(n_iter):
        # perform least squares fit for LV model
        model_approx, status = mp.lstsq_laplace_factor_approx(
            model_approx,
            LV
        )
        opt.model_approx = model_approx

        # perform laplace non linear fit for other factors
        for factor in factors:
            model_approx, status = opt.laplace_factor_approx(
                factor
            )
            history[i, factor] = model_approx

    model_mean = {v.name: d.mean for v, d in model_approx.approx.items()}
    y_pred = LV_model(**model_mean).deterministic_values[y_]
    assert y_pred.min() > 0.02
    assert y_pred.max() < 4
