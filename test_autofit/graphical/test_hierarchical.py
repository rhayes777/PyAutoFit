import numpy as np
from networkx.drawing.tests.test_pylab import plt
from scipy import stats


def normal_loglike(x, centre, precision, _variables=None):
    diff = np.asanyarray(x) - centre
    se = np.square(diff)
    loglike = np.sum(-precision * se - np.log(2 * np.pi) + np.log(precision)) / 2
    if _variables is not None:
        grad = ()
        for v in _variables:
            if v == 'x':
                grad += -precision * diff,
            elif v == 'centre':
                grad += np.sum(precision * diff),
            elif v == 'precision':
                grad += np.sum(1 / precision - se) / 2,

        return loglike, grad
    return loglike


def normal_loglike_t(x, centre, precision, _variables=None):
    # Make log transform of precision so that optimisation is unbounded
    _precision = np.exp(precision)
    val = normal_loglike(
        x, centre, _precision, _variables=_variables)
    if _variables is not None:
        loglike, grad = val
        grad = tuple(
            g * _precision if v == 'precision' else g
            for v, g in zip(_variables, grad)
        )
        return loglike, grad
    return val


def generate_data():
    mu = .5
    sigma = 0.5
    centre_dist = stats.norm(loc=mu, scale=sigma)

    a = 10
    b = 4000
    precision_dist = stats.invgamma(a, scale=b)

    n = 10

    centres = centre_dist.rvs(n)
    widths = precision_dist.rvs(n) ** -0.5

    mu0 = centres.mean()
    lambda0 = n
    alpha0 = n / 2
    beta0 = centres.var() / 2 * n

    # Marginal Posterior Distributions from Normal-Gamma distributions
    posterior_mean = stats.t(
        alpha0 * 2, loc=mu0,
        scale=np.sqrt(beta0 / alpha0 / lambda0))
    posterior_x = stats.t(
        alpha0 * 2, loc=mu0,
        scale=np.sqrt(beta0 / alpha0))
    posterior_t = stats.gamma(alpha0, scale=1 / beta0)

    t_mode = (alpha0 - 1) / beta0
    t_cov = beta0 ** 2 / (alpha0 - 1)

    x = np.linspace(-1.5, 2.5, 200)
    f, (ax1, ax2) = plt.subplots(2)

    for c, w in zip(centres, widths):
        ax1.plot(x, stats.norm(c, scale=w).pdf(x))

    plt.plot(x, posterior_mean.pdf(x), label='Pr(mu|D)')
    plt.plot(x, posterior_x.pdf(x), label='Pr(x|D)')
    # plt.plot(x, posterior_t.pdf(x), label='Pr(t|D)')
    plt.plot(x, stats.norm(loc=mu, scale=sigma).pdf(x), label='Pr(x)')
    plt.legend()


def test():
    generate_data()
