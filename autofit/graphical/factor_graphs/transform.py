

from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.linalg import cho_factor, solve_triangular, get_blas_funcs
from scipy._lib._util import _asarray_validated

from autofit.graphical.factor_graphs import \
    AbstractNode, Variable, Value, FactorValue, JacobianValue, HessianValue
from autofit.graphical.utils import cached_property, Axis, FlattenArrays

class AbstractLinearTransform(ABC):
    @abstractmethod
    def __mul__(self, x:np.ndarray) -> np.ndarray:
        pass 

    @abstractmethod
    def __rtruediv__(self, x:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __rmul__(self, x:np.ndarray) -> np.ndarray:
        pass 

    @abstractmethod
    def ldiv(self, x: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass 

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def size(self) -> int:
        return np.prod(self.shape, dtype=int)

    @cached_property
    @abstractmethod
    def log_det(self):
        pass
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc is np.multiply:
            return self.__rmul__(inputs[0])
        elif ufunc is np.divide:
            return self.__rtruediv__(inputs[0])
        elif ufunc is np.matmul:
            return self.__rmul__(inputs[0])
        else:    
            return NotImplemented

class IdentityTransform(AbstractLinearTransform):
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


def _wrap_leftop(method):
    @wraps(method)
    def leftmethod(self, x):
        return method(self, np.reshape(x, (len(self), -1))).reshape(x.shape)

    return leftmethod

def _wrap_rightop(method):
    @wraps(method)
    def rightmethod(self, x):
        return method(self, np.reshape(x, (-1, len(self)))).reshape(x.shape)

    return rightmethod

class CholeskyTransform(AbstractLinearTransform):
    """ This performs the whitening transforms for the passed
    cholesky factor of the Hessian/inverse covariance of the system.

    see https://en.wikipedia.org/wiki/Whitening_transformation

    >>> M = CholeskyTransform(linalg.cho_factor(hess))
    >>> y = M * x
    >>> f, df_dx = func_and_gradient(M.ldiv(y))
    >>> df_dy = df_df * M
    >>> 
    """

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

class CovarianceTransform(CholeskyTransform):
    """In the case where the covariance matrix is passed
    we perform the inverse operations
    """
    __mul__ = CholeskyTransform.__rtruediv__
    __rmul__ = CholeskyTransform.ldiv
    __rtruediv__ = CholeskyTransform.__mul__
    ldiv = CholeskyTransform.__rmul__

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @cached_property
    def log_det(self):
        return - np.sum(np.log(self.U.diagonal()))


class DiagonalTransform(AbstractLinearTransform):
    def __init__(self, scale, inv_scale=None):
        self.scale = scale
        self.inv_scale = 1/scale if inv_scale is None else scale

    @_wrap_leftop
    def __mul__(self, x):
        return self.inv_scale[:, None] * x 

    @_wrap_rightop
    def __rmul__(self, x):
        return x * self.inv_scale

    @_wrap_rightop
    def __rtruediv__(self, x): 
        return x * self.scale

    @_wrap_leftop
    def ldiv(self, x):
        return self.scale[:, None] * x 

    @cached_property
    def log_det(self):
        return np.sum(np.log(self.inv_scale))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @property
    def shape(self):
        return self.scale.shape * 2
    

class VariableTransform:
    """
    """
    def __init__(self, transforms):
        self.transforms = transforms 
        
    def __mul__(self, values: Value) -> Value:
        return {
            k: M * values[k] for k, M in self.transforms.items()} 

    def __rtruediv__(self, values: Value) -> Value:
        return {
            k: values[k] / M for k, M in self.transforms.items()} 

    def __rmul__(self, values: Value) -> Value:
        return {
            k: values[k] * M for k, M in self.transforms.items()} 
         
    def ldiv(self, values: Value) -> Value:
        return {
            k: M.ldiv(values[k]) for k, M in self.transforms.items()} 

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    def quad(self, values):
        return {
            v: H.T if np.ndim(H) else H
            for v, H in (values * self).items()} * self

    def invquad(self, values):
        return {
            v: H.T if np.ndim(H) else H
             for v, H in (values / self).items()} / self

    @cached_property
    def log_det(self):
        return sum(M.log_det for M in self.transforms.values())

    @classmethod
    def from_scales(cls, scales):
        return cls({
            v: DiagonalTransform(scale) for v, scale in scales.items()
        })

    @classmethod
    def from_covariances(cls, covs):
        return cls({
            v: CovarianceTransform(cho_factor(cov))
            for v, cov in covs.items()
        })

    @classmethod
    def from_inv_covariances(cls, inv_covs):
        return cls({
            v: CholeskyTransform(cho_factor(inv_cov))
            for v, inv_cov in inv_covs.items()
        })


class FullCholeskyTransform(VariableTransform):
    def __init__(self, cholesky, param_shapes):
        self.cholesky = cholesky
        self.param_shapes = param_shapes

    @classmethod
    def from_optresult(cls, opt_result):
        param_shapes = opt_result.param_shapes

        cov = opt_result.result.hess_inv
        if not isinstance(cov, np.ndarray):
            # if optimiser is L-BFGS-B then convert
            # implicit hess_inv into dense matrix
            cov = cov.todense()

        return cls(
            CovarianceTransform.from_dense(cov),
            param_shapes)

    def __mul__(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(M * x)

    def __rtruediv__(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(x / M)

    def __rmul__(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(x * M)
         
    @abstractmethod
    def ldiv(self, values: Value) -> Value:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(M.ldiv(x))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @cached_property
    def log_det(self):
        return self.cholesky.log_det


class IdentityVariableTransform(VariableTransform):
    def __init__(self):
        pass

    def _identity(self, values: Value) -> Value:
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

identity_transform = IdentityTransform()
identity_variable_transform = IdentityVariableTransform()

class TransformedNode(AbstractNode):
    def __init__(
        self, 
        node: AbstractNode, 
        transform: VariableTransform
    ):
        self.node = node 
        self.transform = transform 

    @property 
    def variables(self):
        return self.node.variables 

    @property
    def deterministic_variables(self):
        return self.node.deterministic_variables

    @property
    def all_variables(self):
        return self.node.all_variables

    @property
    def name(self):
        return f"FactorApproximation({self.node.name})"

    def __call__(
            self, 
            values: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> FactorValue:
        return self.node(self.transform.ldiv(values), axis=axis)

    def func_jacobian(
            self, 
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
            **kwargs, 
    ) -> Tuple[FactorValue, JacobianValue]:
        fval, jval = self.node.func_jacobian(
            self.transform.ldiv(values), 
            variables=variables, 
            axis=axis,
            _calc_deterministic=_calc_deterministic)

        # TODO this doesn't deal with deterministic jacobians
        grad = jval / self.transform
        return fval, grad

    def func_jacobian_hessian(
            self, 
            values: Dict[Variable, np.ndarray],
            variables: Optional[List[Variable]] = None,
            axis: Axis = None,
            _calc_deterministic: bool = True,
            **kwargs, 
    ) -> Tuple[FactorValue, JacobianValue, HessianValue]:
        M = self.transform
        fval, jval, hval = self.node.func_jacobian_hessian(
            M.ldiv(values), 
            variables=variables, 
            axis=axis,
            _calc_deterministic=_calc_deterministic)

        grad = jval / M
        # hess = {v: H.T for v, H in (hval / M).items()} / M
        hess = M.invquad(hval)
        return fval, grad, hess

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.node, name)
