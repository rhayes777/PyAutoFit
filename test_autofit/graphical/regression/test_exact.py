
from typing import Tuple 

import numpy as np
from scipy.stats import norm

from autofit import graphical as graph 
from autofit.messages import NormalMessage


phi = norm.pdf 
probit = Phi = norm.cdf
logPhi = norm.logcdf
inv_probit = norm.ppf

np.random.seed(1)

prior_std = 10.0
error_std = 1.0
n_obs = 100

coefs = [-1.3, 0.7, 0.5]

n_features = len(coefs) - 1 # fitting constance

x = 5 * np.random.randn(n_obs, n_features)
X = np.c_[x, np.ones(n_obs)]
z = X @ coefs + np.random.randn(n_obs)

p = probit(z)
y = np.random.binomial(1, p)


def probitnormalprojection(a, b, mu, sigma):
    t = 1./np.sqrt(1 + b**2 * sigma**2)
    bmu = b * mu
    bs2 = b * sigma**2
    z = (a + bmu) * t
    
    phi_z = phi(z)
    Phi_z = Phi(z)
    
    I0 = Phi_z
    I1 = bs2 * t * phi_z + mu * I0
    I2 = (
        bs2 * (2*mu + bs2 * (bmu - a)) * t**3 * phi_z
        + (mu**2 + sigma**2) * I0
    )
    m_proj = I1 / I0 
    var_proj = I2 / I0 - m_proj**2
    return I0, m_proj, var_proj

def probit_posterior(y, mu, sigma):
    b = np.where(y == 1, 1, -1)
    return probitnormalprojection(1, b, mu, sigma)


class ProbitModel:
    __name__ = "ProbitModel"

    def __init__(self, y):
        self.y = y

    def __call__(self, z):
        p = probit(z)
        # Cross entropy
        return np.where(self.y == 1, np.log(p), np.log1p(-p)).sum()

    @staticmethod
    def has_exact_projection(dist):
        return isinstance(dist, NormalMessage)

    def calc_exact_projection(self, dist: NormalMessage) -> Tuple[NormalMessage]:
        I0, m_proj, var_proj = probit_posterior(self.y, dist.mean, dist.sigma)
        return dist.from_mode(m_proj, var_proj, log_norm=np.log(I0).sum()), 

class LinearModel:
    __name__ = "LinearModel"

    def __init__(self, x):
        self.X = np.asanyarray(x)

    def __call__(self, coefs):
        return self.X @ coefs

    @staticmethod
    def has_exact_projection(*args):
        return all(isinstance(dist, NormalMessage) for dist in args)

    def calc_exact_projection(
        self, coefs_dist: NormalMessage, y_dist: NormalMessage
    ) -> Tuple[NormalMessage, NormalMessage]:
        coefs = coefs_dist.mean 
        coefs_var = coefs_dist.variance 
        coefs_prec = coefs_var**-2
        y = y_dist.mean 
        y_var = y_dist.variance 

        X = self.X
        w = y_var**-0.5 
    
        Xp = X * w[:, None]
        yp = y * w 

        Lam_coef = Xp.T.dot(Xp) + np.diag(coefs_prec)
        Sigma_coef = np.linalg.inv(Lam_coef)

        mu_coef = Sigma_coef.dot(yp.dot(Xp) + coefs_prec * coefs)
        
        Lam_ycoef = 1 / np.einsum("jk,ij,ik->i", Sigma_coef, X, X)
        Lam_y = w**2 + Lam_ycoef
        mu_y = (w**2 * y + Lam_ycoef * X.dot(mu_coef)) / Lam_y

        return (
            coefs_dist.from_mode(mu_coef, Sigma_coef.diagonal()), 
            y_dist.from_mode(mu_y, 1/Lam_y)
        )
    

def make_model():
    linear_model = LinearModel(X)
    probit_model = ProbitModel(y)

    f_dist = graph.messages.NormalMessage.from_mode(
        np.zeros(n_obs), 100.
    )
    coef_dist = graph.messages.NormalMessage.from_mode(
        np.zeros(n_features + 1), 100.
    )

    f_, coef_ = graph.variables("f, coef")

    linear_factor = graph.Factor(linear_model, coef_, factor_out=f_)
    probit_factor = graph.Factor(probit_model, f_)
    model = probit_factor * linear_factor 

    model_approx = graph.EPMeanField.from_approx_dists(
        model, {f_: f_dist, coef_: coef_dist}
    )

    return model_approx 

def test_factor_approx():
    model_approx = make_model()
    for factor in model_approx.factors:
        factor_approx = model_approx.factor_approximation(factor)
        assert factor_approx.mean_field.keys() == factor.all_variables

def test_probit_regression():
    model_approx = make_model()
    ep_opt = graph.EPOptimiser.from_meanfield(model_approx)

    fit_approx = ep_opt.run(model_approx)

    f = fit_approx.mean_field.names['f'].mean
    y_pred = probit(f) > 0.5
    acc = (y_pred == y).mean()
    assert acc > 0.9


def test_probit_regression_no_path():
    model_approx = make_model()
    
    # testing paths=False codepath
    ep_opt = graph.EPOptimiser.from_meanfield(model_approx, paths=False)

    fit_approx = ep_opt.run(model_approx)

    f = fit_approx.mean_field.names['f'].mean
    y_pred = probit(f) > 0.5
    acc = (y_pred == y).mean()
    assert acc > 0.9
