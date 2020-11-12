
import numpy as np

import pytest
import numpy as np
from scipy import stats, linalg

import autofit.graphical as graph
import autofit.graphical.factor_graphs.transform as transform



def test_un_whiten_cholesky():
    d = 4
    A = stats.wishart(d, np.eye(d)).rvs()
    x = np.random.rand(d)
    X = np.random.rand(d, d)
    U, _ = linalg.cho_factor(A, lower=False)
    L, _ = linalg.cho_factor(A, lower=True)

    U = np.triu(U, 0) # clear lower diagonal
    L = np.tril(L, 0) # clear upper diagonal

    assert pytest.approx(0) == np.linalg.norm(U - L.T)
    assert pytest.approx(0) == np.linalg.norm(U.T.dot(U) - A)
    assert pytest.approx(0) == (np.linalg.norm(L.dot(L.T) - A))
    assert pytest.approx(0) == (np.linalg.norm(L.dot(U) - A))
    assert pytest.approx(0) == (np.linalg.norm(
        transform._mul_triangular((L, True), x, trans=True) 
        - transform._mul_triangular((U, False), x, trans=False)))
    
    # test unwhiten
    assert pytest.approx(0) == (np.linalg.norm(
        U.dot(x) 
        - transform._unwhiten_cholesky((L, True), x)))
    assert pytest.approx(0) == (np.linalg.norm(
        L.T.dot(x) 
        - transform._unwhiten_cholesky((U, False), x)))
    assert pytest.approx(0) == (np.linalg.norm(
        U.dot(X).dot(L)
        - transform._unwhiten_cholesky(
            (U, False), transform._unwhiten_cholesky((U, False), X).T).T))
    assert pytest.approx(0) == (np.linalg.norm(
        U.dot(X).dot(L)
        - transform._unwhiten_cholesky(
            (L, True), transform._unwhiten_cholesky((L, True), X).T).T))

    # test whiten
    assert pytest.approx(0) == (np.linalg.norm(
        linalg.solve(U, x)
        - transform._whiten_cholesky((L, True), x)))
    assert pytest.approx(0) == (np.linalg.norm(
        linalg.solve(L.T, x)
        - transform._whiten_cholesky((U, False), x)))
    assert pytest.approx(0) == (np.linalg.norm(
        np.linalg.inv(U).dot(X).dot(np.linalg.inv(L))
        - transform._whiten_cholesky(
            (U, False), transform._whiten_cholesky((U, False), X).T).T))
    assert pytest.approx(0) == (np.linalg.norm(
        linalg.solve(L, linalg.solve(U, X).T, transposed=True).T
        - transform._whiten_cholesky(
            (L, True), transform._whiten_cholesky((L, True), X).T).T))
