
import numpy as np

import pytest
import numpy as np
from scipy import stats, linalg, optimize

import autofit.graphical as graph
import autofit.graphical.factor_graphs.transform as transform

def test_CholeskyTransform():
    d = 10
    A = stats.wishart(d, np.eye(d)).rvs()

    cho_factor = transform.CholeskyTransform(linalg.cho_factor(A))

    U = np.triu(cho_factor.U)
    iU = np.linalg.inv(U)

    b = np.random.rand(d)
    assert np.allclose(cho_factor * b, U @ b)
    assert np.allclose(b * cho_factor, b @ U)
    assert np.allclose(cho_factor.ldiv(b), iU @ b)
    assert np.allclose(b / cho_factor, b @ iU)
    
    b = np.random.rand(d, d)
    assert np.allclose(cho_factor * b, U @ b)
    assert np.allclose(b * cho_factor, b @ U)
    assert np.allclose(cho_factor.ldiv(b), iU @ b)
    assert np.allclose(b / cho_factor, b @ iU)
    
    b = np.random.rand(d, d + 1)
    assert np.allclose(cho_factor * b, U @ b)
    assert np.allclose(cho_factor.ldiv(b), iU @ b)

    b = np.random.rand(d + 1, d)
    assert np.allclose(b * cho_factor, b @ U)
    assert np.allclose(b / cho_factor, b @ iU)

def test_DiagonalTransform():
    d = 3

    scale = np.random.rand(d)
    D = np.diag(scale**-1)
    iD = np.diag(scale)
    diag_scale = transform.DiagonalTransform(scale)

    b = np.random.rand(d)
    assert np.allclose(diag_scale * b, D @ b)
    assert np.allclose(b * diag_scale, b @ D)
    assert np.allclose(diag_scale.ldiv(b), iD @ b)
    assert np.allclose(b / diag_scale, b @ iD)

    b = np.random.rand(d, d)
    assert np.allclose(diag_scale * b, D @ b)
    assert np.allclose(b * diag_scale, b @ D)
    assert np.allclose(diag_scale.ldiv(b), iD @ b)
    assert np.allclose(b / diag_scale, b @ iD)
    
    b = np.random.rand(d, d + 1)
    assert np.allclose(diag_scale * b, D @ b)
    assert np.allclose(diag_scale.ldiv(b), iD @ b)

    b = np.random.rand(d + 1, d)
    assert np.allclose(b * diag_scale, b @ D)
    assert np.allclose(b / diag_scale, b @ iD)


def test_simple_transform_cholesky():

    d = 5
    A = stats.wishart(d, np.eye(d)).rvs()
    b = np.random.rand(d)

    def likelihood(x):
        x = x - b
        return 0.5 * np.linalg.multi_dot((x, A, x))

    x = graph.Variable('x', graph.Plate())
    x0 = np.random.randn(d)

    factor = graph.Factor(likelihood, x=x, is_scalar=True)
    param_shapes = graph.utils.FlattenArrays({x: (d,)})
    func = factor.flatten(param_shapes)

    res = optimize.minimize(func, x0)
    assert np.allclose(res.x, b, rtol=1e-2)
    H, iA = res.hess_inv, np.linalg.inv(A)
    # check R2 score
    assert 1 - np.square(H - iA).mean()/np.square(iA).mean() > 0.95
    
    cho = transform.CholeskyTransform(linalg.cho_factor(A))
    whiten = transform.VariableTransform({x: cho})
    white_factor = transform.TransformedNode(factor, whiten)
    white_func = white_factor.flatten(param_shapes)

    y0 = cho * x0

    res = optimize.minimize(white_func, y0)
    assert np.allclose(res.x, cho * b, atol=1e-3, rtol=1e-3)
    assert np.allclose(res.hess_inv, np.eye(d), atol=1e-6, rtol=1e-5)

    # testing gradients

    grad = white_func.jacobian(y0)
    ngrad = optimize.approx_fprime(y0, white_func, 1e-6)
    assert np.allclose(grad, ngrad, atol=1e-6, rtol=1e-5)

    whiten = transform.FullCholeskyTransform(cho, param_shapes)
    white_factor = transform.TransformedNode(factor, whiten)
    white_func = white_factor.flatten(param_shapes)

    y0 = cho * x0

    res = optimize.minimize(white_func, y0)
    assert np.allclose(res.x, cho * b)
    assert np.allclose(res.hess_inv, np.eye(d), atol=1e-6, rtol=1e-5)

    # testing gradients

    grad = white_func.jacobian(y0)
    ngrad = optimize.approx_fprime(y0, white_func, 1e-6)
    assert np.allclose(grad, ngrad, atol=1e-6, rtol=1e-5)


def test_simple_transform_diagonal():
    # testing DiagonalTransform
    d = 5
    scale = np.random.exponential(size=d)
    A = np.diag(scale**-1)
    b = np.random.randn(d)
    
    x = graph.Variable('x', graph.Plate())
    x0 = np.random.randn(d)
    param_shapes = graph.utils.FlattenArrays({x: (d,)})

    def likelihood(x):
        x = x - b
        return 0.5 * np.linalg.multi_dot((x, A, x))
    
    factor = graph.Factor(likelihood, x=x, is_scalar=True)
    func = factor.flatten(param_shapes)
    
    res = optimize.minimize(func, x0)
    assert np.allclose(res.x, b, rtol=1e-2)
    H, iA = res.hess_inv, np.linalg.inv(A)
    # check R2 score
    assert 1 - np.square(H - iA).mean()/np.square(iA).mean() > 0.95
    
    
    scale = np.random.exponential(size=d)
    A = np.diag(scale**-2)
    
    diag = transform.DiagonalTransform(scale)
    whiten = transform.VariableTransform({x: diag})
    white_factor = transform.TransformedNode(factor, whiten)
    white_func = white_factor.flatten(param_shapes)

    y0 = diag * x0

    res = optimize.minimize(white_func, y0)
    assert np.allclose(res.x, diag * b)
    H, iA = res.hess_inv, np.eye(d)
    # check R2 score
    assert 1 - np.square(H - iA).mean()/np.square(iA).mean() > 0.95
    
    # testing gradients
    grad = white_func.jacobian(y0)
    ngrad = optimize.approx_fprime(y0, white_func, 1e-6)
    assert np.allclose(grad, ngrad, atol=1e-6, rtol=1e-4)
