import numpy as np
import pytest
from scipy import stats, integrate

import numpy as np

import autofit.graphical.messages.normal
from autofit import graphical as graph


def check_dist_norm(dist):
    norm, err = integrate.quad(dist.pdf, *dist._support[0])
    assert norm == pytest.approx(1, abs=err)


def check_dist_norms(dist):
    vals, err = integrate.quad_vec(
        lambda x: dist.pdf(np.full(dist.shape, x)),
        *dist._support[0]
    )
    assert np.allclose(vals, 1, atol=err)


def check_log_normalisation(ms):
    m1, *m2s = ms
    A = np.exp(m1.log_normalisation(*m2s))

    # Calculate normalisation numerically
    i12, ierr = integrate.quad(
        lambda x: np.exp(sum(m.logpdf(x) for m in ms)),
        *m1._support[0])

    # verify within tolerance
    assert np.abs(A - i12) < ierr < 1e-6


def check_numerical_gradient_hessians(message, x):
    res = message.logpdf_gradient_hessian(x)
    nres = message.numerical_logpdf_gradient_hessian(x)

    assert np.allclose(res, nres, atol=1e-2)


def test_message_norm():
    messages = [
        tuple(
            map(graph.NormalMessage,
                [0.5, 0.1], [0.2, 0.3])),
        tuple(
            map(graph.NormalMessage,
                [0.5, 0.1, -0.5], [0.2, 0.3, 1.3])),
        # tuple(
        #     map(graph.UniformNormalMessage,
        #         [0.5, 0.1, -0.5], [0.2, 0.3, 1.3])),
        tuple(
            map(graph.GammaMessage,
                [0.5, 1.1], [0.2, 1.3])),
        tuple(
            map(graph.GammaMessage,
                [0.5, 1.1, 2], [0.2, 1.3, 1])),
        tuple(
            map(graph.BetaMessage,
                [2., 3.2, 1.5], [4.1, 2.3, 3])),
    ]
    for ms in messages:
        check_log_normalisation(ms)
        for m in ms:
            check_dist_norm(m)


def test_numerical_gradient_hessians():
    N = graph.NormalMessage
    test_cases = [
        (N, 1., 0.5, 0.3),
        (N, 1., 0.5, [0.3, 2.1]),
        (N, [0.1, 1., 2.], [2., 0.5, 3.], [0.1, 0.2, 0.3]),
        (N, [0.1, 1., 2.], [2., 0.5, 3.], [[0.1, 0.2, 0.3], [2., 1., -1]]),
    ]
    for M, m, s, x in test_cases:
        check_numerical_gradient_hessians(M(m, s), x)


def test_meanfield_gradients():
    n1, n2, n3 = 2, 3, 5
    p1, p2, p3 = [graph.Plate() for i in range(3)]

    v1 = graph.Variable('v1', p1, p2)
    v2 = graph.Variable('v2', p2, p3)
    v3 = graph.Variable('v3', p3, p1)

    mean_field = graph.MeanField({
        v1: graph.NormalMessage(
            np.random.randn(n1, n2),
            np.random.exponential(size=(n1, n2))),
        v2: graph.NormalMessage(
            np.random.randn(n2, n3),
            np.random.exponential(size=(n2, n3))),
        v3: graph.NormalMessage(
            np.random.randn(n3, n1),
            np.random.exponential(size=(n3, n1)))})

    values = mean_field.sample()
    l0 = mean_field(values, axis=None)
    logl = mean_field(values, axis=False)
    assert logl.sum() == pytest.approx(l0, abs=1e-5)
    logl = mean_field(values, axis=1)
    assert logl.sum() == pytest.approx(l0, abs=1e-5)
    logl = mean_field(values, axis=2)
    assert logl.sum() == pytest.approx(l0, abs=1e-5)
    logl = mean_field(values, axis=(0, 2))
    assert logl.sum() == pytest.approx(l0, abs=1e-5)

    njac0 = mean_field._numerical_func_jacobian(
        values, axis=None, _eps=1e-8)[1]
    njac1 = mean_field._numerical_func_jacobian(
        values, axis=1, _eps=1e-8)[1]
    njac2 = mean_field._numerical_func_jacobian(
        values, axis=(0, 1), _eps=1e-8)[1]
    njac = mean_field._numerical_func_jacobian_hessian(
        values, axis=False, _eps=1e-8)[1]
    grad = mean_field.logpdf_gradient(values, axis=False)[1]
    for v in grad:
        norm = np.linalg.norm(grad[v] - njac[v].sum((0, 1, 2)))
        assert norm == pytest.approx(0, abs=1e-2)
        norm = np.linalg.norm(grad[v] - njac0[v])
        assert norm == pytest.approx(0, abs=1e-2)
        norm = np.linalg.norm(grad[v] - njac1[v].sum((0, 1)))
        assert norm == pytest.approx(0, abs=1e-2)
        norm = np.linalg.norm(grad[v] - njac2[v].sum(0))
        assert norm == pytest.approx(0, abs=1e-2)


def test_beta():
    a = b = np.r_[2., 3.2, 1.5]
    beta = graph.BetaMessage(a, b[::-1])
    check_dist_norms(beta)

    betas = [
        graph.BetaMessage(a, b)
        for a, b in (np.random.poisson(5, size=(10, 2)) + 1)
    ]
    [check_dist_norm(b) for b in betas]
    check_log_normalisation(betas)


def test_transforms():
    a = b = np.r_[2., 3.2, 1.5]
    mx, sigma = np.r_[-0.5, .1, 0.], np.r_[0.3, 0.5, 0.9]
    test_cases = [
        (graph.BetaMessage, (a, b)),
        (graph.NormalMessage, (mx, sigma)),
        (graph.UniformNormalMessage, (mx, sigma)),
    ]
    shift = (0.3, 5.1)
    beta = graph.BetaMessage.shifted(0.3, 5.1)(a, b[::-1])


