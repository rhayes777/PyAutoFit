
import numpy as np

import pytest
import numpy as np
from scipy import stats, linalg, optimize

import autofit.graphical as graph
import autofit.graphical.factor_graphs.transform as transform

def test_cholesky_transform():
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

def test_diagonal_transform():
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

    np.random.seed(0)

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
    
    # cho = transform.CholeskyTransform(linalg.cho_factor(A))
    cho = transform.CholeskyTransform.from_dense(A)
    whiten = transform.VariableTransform({x: cho})
    white_factor = transform.TransformedNode(factor, whiten)
    white_func = white_factor.flatten(param_shapes)

    y0 = cho * x0

    res = optimize.minimize(white_func, y0)
    assert np.allclose(res.x, cho * b, atol=1e-3, rtol=1e-3)
    assert np.allclose(res.hess_inv, np.eye(d), atol=1e-3, rtol=1e-3)

    # testing gradients

    grad = white_func.jacobian(y0)
    ngrad = optimize.approx_fprime(y0, white_func, 1e-6)
    assert np.allclose(grad, ngrad, atol=1e-3, rtol=1e-3)

    # testing CovarianceTransform,

    cho = transform.CovarianceTransform.from_dense(iA)
    whiten = transform.VariableTransform({x: cho})
    white_factor = transform.TransformedNode(factor, whiten)
    white_func = white_factor.flatten(param_shapes)

    y0 = cho * x0

    res = optimize.minimize(white_func, y0)
    assert np.allclose(res.x, cho * b, atol=1e-3, rtol=1e-3)
    assert np.allclose(res.hess_inv, np.eye(d), atol=1e-3, rtol=1e-3)

    # testing gradients

    grad = white_func.jacobian(y0)
    ngrad = optimize.approx_fprime(y0, white_func, 1e-6)
    assert np.allclose(grad, ngrad, atol=1e-3, rtol=1e-3)


    # testing FullCholeskyTransform

    whiten = transform.FullCholeskyTransform(cho, param_shapes)
    white_factor = transform.TransformedNode(factor, whiten)
    white_func = white_factor.flatten(param_shapes)

    y0 = cho * x0

    res = optimize.minimize(white_func, y0)
    assert np.allclose(res.x, cho * b, atol=1e-3, rtol=1e-3)
    assert np.allclose(res.hess_inv, np.eye(d), atol=1e-3, rtol=1e-3)

    # testing gradients

    grad = white_func.jacobian(y0)
    ngrad = optimize.approx_fprime(y0, white_func, 1e-6)
    assert np.allclose(grad, ngrad, atol=1e-3, rtol=1e-3)


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
    assert np.allclose(res.x, b, rtol=1e-2, atol=1e-2)
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
    assert np.allclose(grad, ngrad, atol=1e-3, rtol=1e-3)


def test_complex_transform():

    n1, n2, n3 = 2, 3, 2
    d = n1 + n2 * n3

    A = stats.wishart(d, np.eye(d)).rvs()
    b = np.random.rand(d)

    p1, p2, p3 = (graph.Plate() for i in range(3))
    x1 = graph.Variable('x1', p1)
    x2 = graph.Variable('x2', p2, p3)

    mean_field = graph.MeanField({
        x1: graph.NormalMessage(np.zeros(n1),100*np.ones(n1)),
        x2: graph.NormalMessage(np.zeros((n2, n3)),100*np.ones((n2, n3))),
    })

    values = mean_field.sample()
    param_shapes = graph.utils.FlattenArrays(
        {v: x.shape for v, x in values.items()})

    def likelihood(x1, x2):
        x = np.r_[x1, x2.ravel()] - b
        return 0.5 * np.linalg.multi_dot((x, A, x))

    factor = graph.Factor(likelihood, x1=x1, x2=x2, is_scalar=True)

    cho = transform.CholeskyTransform(linalg.cho_factor(A))
    whiten = transform.FullCholeskyTransform(cho, param_shapes)
    trans_factor = transform.TransformedNode(factor, whiten)

    values = mean_field.sample()
    transformed = whiten * values

    assert np.allclose(factor(values), trans_factor(transformed))

    njac = trans_factor._numerical_func_jacobian(transformed)[1]
    jac = trans_factor.jacobian(transformed)
    ngrad = param_shapes.flatten(njac)
    grad = param_shapes.flatten(jac)

    assert np.allclose(grad, ngrad, atol=1e-3, rtol=1e-3)

    # test VariableTransform with CholeskyTransform
    var_cov = {
        v: (X.reshape((int(X.size**0.5),)*2))
        for v, X in param_shapes.unflatten(linalg.inv(A)).items()
    }
    cho_factors = {
        v:  transform.CholeskyTransform(
            linalg.cho_factor(linalg.inv(cov)))
        for v, cov in var_cov.items()
    }
    whiten = transform.VariableTransform(cho_factors)
    trans_factor = transform.TransformedNode(factor, whiten)

    values = mean_field.sample()
    transformed = whiten * values

    assert np.allclose(factor(values), trans_factor(transformed))

    njac = trans_factor._numerical_func_jacobian(transformed)[1]
    jac = trans_factor.jacobian(transformed)
    ngrad = param_shapes.flatten(njac)
    grad = param_shapes.flatten(jac)

    assert np.allclose(grad, ngrad)

    res = optimize.minimize(
        trans_factor.flatten(param_shapes).func_jacobian, 
        param_shapes.flatten(transformed),
        method='BFGS', jac=True
    )
    assert res.hess_inv.diagonal() == pytest.approx(1., rel=1e-1)

    # test VariableTransform with CholeskyTransform
    diag_factors = {
        v: transform.DiagonalTransform(cov.diagonal()**0.5)
        for v, cov in var_cov.items()
    }
    whiten = transform.VariableTransform(diag_factors)
    trans_factor = transform.TransformedNode(factor, whiten)

    values = mean_field.sample()
    transformed = whiten * values

    assert np.allclose(factor(values), trans_factor(transformed))

    njac = trans_factor._numerical_func_jacobian(transformed)[1]
    jac = trans_factor.jacobian(transformed)
    ngrad = param_shapes.flatten(njac)
    grad = param_shapes.flatten(jac)

    assert np.allclose(grad, ngrad, atol=1e-3, rtol=1e-3)

    res = optimize.minimize(
        trans_factor.flatten(param_shapes).func_jacobian, 
        param_shapes.flatten(transformed),
        method='BFGS', jac=True
    )
    assert res.hess_inv.diagonal() == pytest.approx(1., rel=1e-1)