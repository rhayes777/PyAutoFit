from abc import ABC, abstractmethod
from functools import wraps
from typing import Tuple, List

import numpy as np
from scipy._lib._util import _asarray_validated
from scipy.linalg import cho_factor, solve_triangular, get_blas_funcs, block_diag

from autoconf import cached_property


class LinearOperator(ABC):
    """Implements the functionality of a linear operator. 

    All linear operators can be expressed as a tensor/matrix
    However for some it there may be more efficient representations

    e.g. The Diagonal Matrix (see `DiagonalMatrix`), 
    Sherman-Morrison Matrix (see `ShermanMorrison`)
    or the vector-Jacobian product of a function see jax.jvp [0]

    The class also has the attributes, lshape, rshape, lsize, rsize to allow multidimensional tensors to be used,
    see `ShermanMorrison`, `MultiVecOuterProduct`, or 
    `autofit.messages.transform.MultinomialLogitTransform`
    for examples of this use case

    If `M` is the dense matrix represenation of the LinearOperator then the
    actions of the methods can be represented by the appropriate matrix
    operation, 

    Methods
    -------
    __mul__(x):
        M.dot(x) 
        M·x

    __rmul__(x):
        x.dot(M) 
        x·M

    __rtruediv__(x):
        x.dot(inv(M)) 
        x / M

    ldiv(x):
        inv(M).dot(x) 
        M \ x

    log_det():
        log(det(M))
        log |M|

    quad(x):
        x.T.dot(M).dot(x)
        xᵀ·M·x

    invquad(x):
        x.T.dot(inv(M)).dot(x)
        xᵀ·M⁻¹·x

    [0] https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-product
    """
    _ldim = 1

    @abstractmethod
    def __mul__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __rtruediv__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __rmul__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def ldiv(self, x: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    def __len__(self) -> int:
        return self.lsize

    @property
    def lsize(self) -> int:
        return np.prod(self.lshape, dtype=int)

    @property
    def rsize(self) -> int:
        return np.prod(self.rshape, dtype=int)

    @property
    def size(self) -> int:
        return np.prod(self.shape, dtype=int)

    @property
    def ldim(self):
        return self._ldim

    @property
    def rdim(self):
        return self.ndim - self.ldim

    def ndim(self):
        return len(self.shape)

    @property
    def lshape(self):
        return self.shape[:self.ldim]

    @property
    def rshape(self):
        return self.shape[self.ldim:]

    @cached_property
    @abstractmethod
    def log_det(self):
        pass

    def quad(self, M: np.ndarray) -> np.ndarray:
        return (M * self).T * self

    def invquad(self, M: np.ndarray) -> np.ndarray:
        return (M / self).T / self

    def transform_bounds(
            self,
            bounds: List[Tuple]
    ) -> List[Tuple]:
        """
        Convenience method for transforming the bounds of an
        operation by this matrix.

        Parameters
        ----------
        bounds
            A list of tuples, each describing the lower and upper
            bound for a given dimension.

            There should be N bounds corresponding to N dimensions
            for this NxN matrix.

        Returns
        -------
        The bounds transformed according to this transformation.
        """
        lower, upper = zip(
            *bounds
        )
        lower = self * lower
        upper = self * upper
        return list(
            zip(
                lower,
                upper
            )
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in (np.multiply, np.matmul):
            return self.__rmul__(inputs[0])
        elif ufunc is np.divide:
            return self.__rtruediv__(inputs[0])
        else:
            return NotImplemented


class MatrixOperator(LinearOperator):

    def __init__(self, M: np.ndarray, shape=None, ldim=1):
        self.M = np.asanyarray(M)
        self._shape = shape or self.M.shape
        self._ldim = 1

    @property
    def shape(self):
        return self._shape

    @cached_property
    def log_det(self):
        return np.linalg.slogdet(self.M)[1]

    def __mul__(self, x: np.ndarray) -> np.ndarray:
        return self.M.dot(x)

    def __rtruediv__(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.M.T, x.T)

    def __rmul__(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.M)

    def ldiv(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.M, x)

    def to_dense(self):
        return self.M


class IdentityOperator(LinearOperator):
    def __init__(self):
        pass

    def _identity(self, values: np.ndarray) -> np.ndarray:
        return values

    __mul__ = _identity
    __rtruediv__ = _identity
    __rmul__ = _identity
    ldiv = _identity
    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    quad = _identity
    invquad = _identity

    @property
    def log_det(self):
        return 0.

    @property
    def shape(self):
        return ()

    def __len__(self):
        return 0


def _mul_triangular(c, b, trans=False, lower=True, overwrite_b=False,
                    check_finite=True):
    """wrapper for BLAS function trmv to perform triangular matrix
    multiplications


    Parameters
    ----------
    a : (M, M) array_like
        A triangular matrix
    b : (M,) or (M, N) array_like
        vector/matrix being multiplied
    lower : bool, optional
        Use only data contained in the lower triangle of `a`.
        Default is to use upper triangle.
    trans : bool, optional
        type of multiplication,

        ========  =========
        trans     system
        ========  =========
        False     a b
        True      a^T b
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance)
        not fully tested
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    """
    a1 = _asarray_validated(c, check_finite=check_finite)
    b1 = _asarray_validated(b, check_finite=check_finite)

    n = c.shape[1]
    if c.shape[0] != n:
        raise ValueError("Triangular matrix passed must be square")
    if b.shape[0] != n:
        raise ValueError(
            f"shapes {c.shape} and {b.shape} not aligned: "
            f"{n} (dim 1) != {b.shape[0]} (dim 0)")

    trmv, = get_blas_funcs(('trmv',), (a1, b1))
    if a1.flags.f_contiguous:
        def _trmv(a1, b1, overwrite_x):
            return trmv(
                a1, b1,
                lower=lower, trans=trans, overwrite_x=overwrite_x)
    else:
        # transposed system is solved since trmv expects Fortran ordering
        def _trmv(a1, b1, overwrite_x=overwrite_b):
            return trmv(
                a1.T, b1,
                lower=not lower, trans=not trans, overwrite_x=overwrite_x)

    if b1.ndim == 1:
        return _trmv(a1, b1, overwrite_b)
    elif b1.ndim == 2:
        # trmv only works for vector multiplications
        # set Fortran order so memory contiguous
        b2 = np.array(b1, order='F')
        for i in range(b2.shape[1]):
            # overwrite results
            _trmv(a1, b2[:, i], True)

        if overwrite_b:
            b1[:] = b2
            return b1
        else:
            return b2
    else:
        raise ValueError("b must have 1 or 2 dimensions, has {b.ndim}")


# TODO refactor these for non-square transformations

def _wrap_leftop(method):
    @wraps(method)
    def leftmethod(self, x: np.ndarray) -> np.ndarray:
        x = np.asanyarray(x)
        return method(self, x.reshape(self.rsize, -1)).reshape(x.shape)

    return leftmethod


def _wrap_rightop(method):
    @wraps(method)
    def rightmethod(self, x: np.ndarray) -> np.ndarray:
        x = np.asanyarray(x)
        return method(self, x.reshape(-1, self.lsize)).reshape(x.shape)

    return rightmethod


class CholeskyOperator(LinearOperator):
    """ This performs the whitening transforms for the passed
    cholesky factor of the Hessian/inverse covariance of the system.

    see https://en.wikipedia.org/wiki/Whitening_transformation

    >>> M = CholeskyTransform(linalg.cho_factor(hess))
    >>> y = M * x
    >>> f, df_dx = func_and_gradient(M.ldiv(y))
    >>> df_dy = df_df * M
    >>> 
    """

    # TODO implement tensor shape functionality

    def __init__(self, cho_factor):
        self.c, self.lower = self.cho_factor = cho_factor
        self.L = self.c if self.lower else self.c.T
        self.U = self.c.T if self.lower else self.c

    @classmethod
    def from_dense(cls, hess):
        return cls(cho_factor(hess))

    @_wrap_leftop
    def __mul__(self, x):
        return _mul_triangular(self.U, x, lower=False)

    @_wrap_rightop
    def __rmul__(self, x):
        return _mul_triangular(self.L, x.T, lower=True).T

    @_wrap_rightop
    def __rtruediv__(self, x):
        return solve_triangular(self.L, x.T, lower=True).T

    @_wrap_leftop
    def ldiv(self, x):
        return solve_triangular(self.U, x, lower=False)

    @cached_property
    def log_det(self):
        return np.sum(np.log(self.U.diagonal()))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @property
    def shape(self):
        return self.c.shape

    def to_dense(self):
        return np.tril(self.L) @ np.triu(self.U)


class InverseLinearOperator(LinearOperator):
    def __init__(self, transform):
        self.transform = transform

    @abstractmethod
    def __mul__(self, x: np.ndarray) -> np.ndarray:
        return x / self.transform

    @abstractmethod
    def __rtruediv__(self, x: np.ndarray) -> np.ndarray:
        return self.transform * x

    @abstractmethod
    def __rmul__(self, x: np.ndarray) -> np.ndarray:
        return self.transform.ldiv(x)

    @abstractmethod
    def ldiv(self, x: np.ndarray) -> np.ndarray:
        return x * self.transform

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.transform.shape

    @property
    def ldim(self):
        return self.transform.ldim

    @cached_property
    def log_det(self):
        return - self.transform.log_det

    def to_dense(self):
        return np.linalg.inv(self.transform.to_dense())


class InvCholeskyTransform(CholeskyOperator):
    """In the case where the covariance matrix is passed
    we perform the inverse operations
    """
    __mul__ = CholeskyOperator.__rtruediv__
    __rmul__ = CholeskyOperator.ldiv
    __rtruediv__ = CholeskyOperator.__mul__
    ldiv = CholeskyOperator.__rmul__

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @cached_property
    def log_det(self):
        return - np.sum(np.log(self.U.diagonal()))

    def to_dense(self):
        return solve_triangular(self.U, np.triul(self.L), lower=False)


class DiagonalMatrix(LinearOperator):
    """
    Represents the DiagonalMatrix with diagonal `scale`

    M = np.diag(scale.ravel())
    
    """

    def __init__(self, scale, inv_scale=None):
        self.scale = np.asanyarray(scale)
        self._fscale = np.ravel(self.scale)
        self._finv_scale = (
            1 / scale
            if inv_scale is None
            else np.ravel(inv_scale)
        )
        self._ldim = len(self.scale.shape)

    @classmethod
    def from_dense(
            cls,
            inverse_hessian: np.ndarray
    ) -> "DiagonalMatrix":
        """
        Create a diagonal matrix from the inverse hessian.

        The matrix transforms parameter space by some coefficient
        in each dimension.

        Parameters
        ----------
        inverse_hessian
            The inverse hessian determined during an optimisation

        Returns
        -------
        A DiagonalMatrix which whitens the parameter space according to
        the hessian
        """
        return cls(
            np.sqrt(
                np.diagonal(
                    inverse_hessian
                )
            )
        )

    @_wrap_leftop
    def __mul__(self, x):
        return self._fscale[:, None] * x

    @_wrap_rightop
    def __rmul__(self, x):
        return x * self._fscale

    @_wrap_rightop
    def __rtruediv__(self, x):
        return x * self._finv_scale

    @_wrap_leftop
    def ldiv(self, x):
        return self._finv_scale[:, None] * x

    @cached_property
    def log_det(self):
        return np.sum(np.log(self.scale))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @property
    def shape(self):
        return self.scale.shape * 2

    def to_dense(self):
        return np.diag(self._fscale).reshape(self.shape)


class VecOuterProduct(LinearOperator):
    """
    represents the matrix vector outer product

    outer = vec[:, None] * vecT[None, :]
    """

    def __init__(self, vec, vecT=None):
        self.vec = np.asanyarray(vec)
        self.vecT = np.asanyarray(vec if vecT is None else vecT)
        self._fvec = np.ravel(self.vec)[:, None]
        self._fvecT = np.ravel(self.vecT)[None, :]

    @_wrap_leftop
    def __mul__(self, x):
        return self._fvec @ self._fvecT.dot(x)

    @_wrap_rightop
    def __rmul__(self, x):
        return x.dot(self._fvec) @ self._fvecT

    def __rtruediv__(self, x):
        raise NotImplementedError()

    def ldiv(self, x):
        raise NotImplementedError()

    @cached_property
    def log_det(self):
        return - np.inf

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    @property
    def shape(self):
        return (self.vec.size, self.vecT.size)

    @property
    def ldim(self):
        return len(self.vec.shape)

    def to_dense(self):
        return np.outer(self._fvec, self._fvecT).reshape(self.shape)


class MultiVecOuterProduct(LinearOperator):
    """
    represents the matrix vector outer product for stacked vectors,

    outer -> block_diag(*
        (v[:, None] * u[None, :] for v, u in zip(vec, vecT)
    )
    outer @ x -> np.vstack([
        v[:, None] * u[None, :] @ x for v, u in zip(vec, vecT)
    ])
    """

    def __init__(self, vec, vecT=None):
        self.vec = np.asanyarray(vec)
        self.vecT = np.asanyarray(vec if vecT is None else vecT)
        self.n, self.d = self.vec.shape
        self.nT, self.dT = self.vecT.shape

    @_wrap_leftop
    def __mul__(self, x):
        return np.einsum(
            "ij,ik,ikl -> ijl",
            self.vec, self.vecT, x.reshape(*self.lshape, -1)
        )

    @_wrap_rightop
    def __rmul__(self, x):
        return np.einsum(
            "ij,ik,lij -> lik",
            self.vec, self.vecT, x.reshape(-1, *self.rshape)
        )

    def __rtruediv__(self, x):
        raise NotImplementedError()

    def ldiv(self, x):
        raise NotImplementedError()

    @cached_property
    def log_det(self):
        return - np.inf

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    @property
    def shape(self):
        return self.vec.shape + self.vecT.shape

    @property
    def lshape(self):
        return self.shape[:2]

    @property
    def rshape(self):
        return self.shape[2:]

    def to_dense(self):
        return block_diag(*(
                self.vec[:, :, None] * self.vecT[:, None, :]
        )).reshape(self.shape)


class ShermanMorrison(LinearOperator):
    """
    Represents the Sherman-Morrison low rank update, 
    inv(A + vec @ vec.T) = 
        inv(A) - inv(A) @ vec @ vec.T @ inv(A) / (1 + vec @ inv(A) @ vec/T)
    """

    def __init__(self, linear, vec):
        self.linear = linear
        if np.ndim(vec) == 2:
            self.outer = MultiVecOuterProduct(vec)
        elif np.ndim(vec) == 1:
            self.outer = VecOuterProduct(vec)
        else:
            raise ValueError("vec must be 1 or 2 dimensional")

        self.inv_scale = 1 + linear.quad(vec)
        self._ldim = self.linear.ldim

    @_wrap_leftop
    def __mul__(self, x):
        return self.linear * x + self.outer * x

    @_wrap_rightop
    def __rmul__(self, x):
        return x * self.linear + x * self.outer

    @_wrap_rightop
    def __rtruediv__(self, x):
        x1 = x / self.linear
        return x1 - ((x1 / self.inv_scale) * self.outer) / self.linear

    @_wrap_leftop
    def ldiv(self, x):
        x1 = self.linear.ldiv(x)
        return x1 - self.linear.ldiv(self.outer.dot(x1 / self.inv_scale))

    @cached_property
    def log_det(self):
        return self.linear.log_det + np.log(self.inv_scale)

    @property
    def shape(self):
        return self.linear.shape

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__
    __rmatmul__ = __rmul__

    def to_dense(self):
        dense_outer = self.outer.to_dense()
        return self.linear.to_dense().reshape(self.shape) + dense_outer
