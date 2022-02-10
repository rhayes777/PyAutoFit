import numpy as np
import pytest
from scipy import stats, linalg

import autofit.graphical.factor_graphs.transform as transform


def test_diagonal_from_dense():
    matrix = transform.DiagonalMatrix.from_dense(
        np.array([[4, 2], [3, 4]])
    ).to_diagonal()

    assert len(matrix.scale) == 2
    assert (matrix.scale == np.array([4, 4])).all()


@pytest.mark.parametrize(
    "bounds, result",
    [
        ([(1, 2), (1, 2)], [(3, 6), (2, 4)]),
        ([(np.inf, np.inf), (np.inf, np.inf)], [(np.inf, np.inf), (np.inf, np.inf)]),
        ([(-2, -1), (0, 2)], [(-6, -3), (0, 4)]),
    ],
)
def test_transform_bounds(bounds, result):
    matrix = transform.DiagonalMatrix(np.array([3, 2]))
    assert matrix.transform_bounds(bounds) == result


def test_cholesky_transform():
    d = 10
    A = stats.wishart(d, np.eye(d)).rvs()

    cho_factor = transform.CholeskyOperator(linalg.cho_factor(A))

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
    D = np.diag(scale)
    iD = np.diag(scale ** -1)
    diag_scale = transform.DiagonalMatrix(scale)

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
