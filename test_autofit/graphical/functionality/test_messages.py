import numpy as np
import pytest
from scipy import integrate

from autofit.messages import transform
from autofit.messages.beta import BetaMessage
from autofit.messages.gamma import GammaMessage
from autofit.messages.normal import (
    NormalMessage,
    UniformNormalMessage,
    LogNormalMessage,
    MultiLogitNormalMessage,
)
from autofit.messages.transform import numerical_jacobian


def check_dist_norm(dist):
    norm, err = integrate.quad(dist.pdf, *dist._support[0])
    assert norm == pytest.approx(1, abs=err), dist


def check_dist_norms(dist):
    vals, err = integrate.quad_vec(
        lambda x: dist.pdf(np.full(dist.shape, x)), *dist._support[0]
    )
    assert np.allclose(vals, 1, atol=err), dist


def check_log_normalisation(ms):
    m1, *m2s = ms
    A = np.exp(m1.log_normalisation(*m2s))

    # Calculate normalisation numerically
    i12, ierr = integrate.quad(
        lambda x: np.exp(sum(m.logpdf(x) for m in ms)), *m1._support[0]
    )

    # verify within tolerance
    assert np.abs(A - i12) < ierr < 1e-1, m1


def check_numerical_gradient_hessians(message, x=None):
    x = message.sample() if x is None else x

    res = message.logpdf_gradient(x)
    nres = message.numerical_logpdf_gradient(x)
    for i, (x1, x2) in enumerate(zip(res, nres)):
        assert np.allclose(x1, x2, rtol=1e-3, atol=1e-2), (i, x1, x2, message)

    res = message.logpdf_gradient_hessian(x)
    nres = message.numerical_logpdf_gradient_hessian(x)
    for i, (x1, x2) in enumerate(zip(res, nres)):
        assert np.allclose(x1, x2, rtol=1e-3, atol=1e-2), (i, x1, x2, message)


def test_message_norm():
    messages = [
        tuple(map(NormalMessage, [0.5, 0.1], [0.2, 0.3])),
        tuple(map(NormalMessage, [0.5, 0.1, -0.5], [0.2, 0.3, 1.3])),
        tuple(map(GammaMessage, [0.5, 1.1], [0.2, 1.3])),
        tuple(map(GammaMessage, [0.5, 1.1, 2], [0.2, 1.3, 1])),
        tuple(map(BetaMessage, [2.0, 3.2, 1.5], [4.1, 2.3, 3])),
    ]
    for ms in messages:
        check_log_normalisation(ms)
        for m in ms:
            check_dist_norm(m)
            check_numerical_gradient_hessians(m)


N = NormalMessage
UN = UniformNormalMessage
SUN = UN.shifted(shift=0.3, scale=0.8)
LN = LogNormalMessage
MLN = MultiLogitNormalMessage
# test doubly transformed distributions
WN = NormalMessage.transformed(transform.log_transform).transformed(
    transform.exp_transform,
)


@pytest.mark.parametrize(
    "M, m, s, x",
    [
        (N, 1.0, 0.5, 0.3),
        (N, 1.0, 0.5, [0.3, 2.1]),
        (N, [0.1, 1.0, 2.0], [2.0, 0.5, 3.0], [0.1, 0.2, 0.3]),
        (N, [0.1, 1.0, 2.0], [2.0, 0.5, 3.0], [[0.1, 0.2, 0.3], [2.0, 1.0, -1]]),
        (UN, 1.0, 0.5, None),
        (UN, [0.1, 1.0, 2.0], [0.1, 0.5, 0.2], None),
        (SUN, 1.0, 0.5, None),
        (SUN, [0.1, 1.0, 2.0], [0.1, 0.5, 0.2], None),
        (LN, 1.0, 0.5, None),
        (LN, [0.1, 1.0, 2.0], [2.0, 0.5, 3.0], None),
        (LN, [0.1, 1.0, 2.0], [2.0, 0.5, 3.0], None),
        (MLN, [0.1, 1.0, 2.0], [0.1, 0.5, 0.2], None),
        (WN, 1.0, 0.5, 0.3),
        (WN, 1.0, 0.5, [0.3, 2.1]),
        (WN, [0.1, 1.0, 2.0], [2.0, 0.5, 3.0], [0.1, 0.2, 0.3]),
        (WN, [0.1, 1.0, 2.0], [2.0, 0.5, 3.0], [[0.1, 0.2, 0.3], [2.0, 1.0, -1]]),
    ],
)
def test_numerical_gradient_hessians(M, m, s, x):
    check_numerical_gradient_hessians(M(m, s), x)


def test_beta():
    a = b = np.r_[2.0, 3.2, 1.5]
    beta = BetaMessage(a, b[::-1])
    check_dist_norms(beta)

    betas = [BetaMessage(a, b) for a, b in (np.random.poisson(5, size=(10, 2)) + 1)]
    for b in betas:
        check_dist_norm(b)

    check_log_normalisation(betas)


def check_transforms(transform, x):
    y, logd, logd_grad, jac = transform.transform_det_jac(x)

    njac = numerical_jacobian(x, transform.transform)
    nlogd_grad = numerical_jacobian(x, transform.log_det)

    assert np.allclose(transform.inv_transform(y), x)
    assert np.allclose(jac.to_dense(), njac)
    assert np.isclose(logd.sum(), np.linalg.slogdet(njac)[1])
    assert np.allclose(nlogd_grad.sum(0), logd_grad)


def test_transforms():
    tests = [
        (transform.log_transform, np.r_[10, 2, 0.1]),
        (transform.exp_transform, np.r_[10, -2, 0.1]),
        (transform.logistic_transform, np.r_[0.22, 0.51, 0.1]),
        (transform.phi_transform, np.r_[0.22, 0.51, 0.1]),
        (transform.shifted_logistic(shift=11, scale=5.1), np.r_[11.1, 12, 16]),
    ]
    for args in tests:
        check_transforms(*args)


def test_multinomial_logit():
    mult_logit = transform.multinomial_logit_transform

    d = 3
    p = np.random.dirichlet(np.ones(d + 1))[:d]

    x, logd, logd_grad, jac = mult_logit.transform_det_jac(p)

    njac = numerical_jacobian(p, mult_logit.transform)
    nlogd_grad = numerical_jacobian(p, mult_logit.log_det)

    assert np.allclose(mult_logit.inv_transform(x), p)
    assert np.allclose(jac.to_dense(), njac)
    assert np.isclose(logd.sum(), np.linalg.slogdet(njac)[1])
    assert np.allclose(nlogd_grad.sum(0), logd_grad)

    n = 5

    ps = np.random.dirichlet(np.ones(d + 1), size=n)[:, :d]
    xs, logd, logd_grad, jac = mult_logit.transform_det_jac(ps)
    njac = numerical_jacobian(ps, mult_logit.transform).reshape(jac.shape)
    nlogd_grad = numerical_jacobian(ps, mult_logit.log_det)

    assert np.allclose(mult_logit.inv_transform(xs), ps)
    assert np.allclose(njac, jac.to_dense())
    assert np.isclose(
        logd.sum(), np.linalg.slogdet(njac.reshape(jac.lsize, jac.rsize))[1]
    )
    assert np.allclose(nlogd_grad.sum((0, 1)), logd_grad, 1e-5, 1e-3)

    assert np.allclose(xs[0], mult_logit.transform(ps[0]))
    assert np.allclose(logd[0], mult_logit.log_det(ps[0]))
    assert np.allclose(logd_grad[0], mult_logit.log_det_grad(ps[0])[1])


def test_normal_simplex():
    mult_logit = transform.MultinomialLogitTransform()
    NormalSimplex = NormalMessage.transformed(mult_logit)

    message = NormalSimplex([-1, 2], [0.3, 0.3])

    check_numerical_gradient_hessians(message, message.sample())

    def func(*p):
        return np.exp(message.factor(p)).prod()

    def simplex_lims(*args):
        return [0, 1 - sum(args)]

    # verify transformation normalises correctly
    res, err = integrate.nquad(func, [simplex_lims] * message.size)
    assert res == pytest.approx(1, rel=err)
