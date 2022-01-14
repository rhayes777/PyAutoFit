
from abc import ABC, abstractmethod
from functools import wraps
from typing import  Dict
import operator

import numpy as np
from scipy.linalg import cho_factor

from autoconf import cached_property

from .operator import (
    LinearOperator, IdentityOperator, DiagonalMatrix,
    CholeskyOperator, MatrixOperator
)
from .variable import Variable


def _unary_op(op):
    @wraps(op)
    def __op__(self):
        return type(self)({k: op(val) for k, val in self.items()})
    
    return __op__

def _binary_op(op):
    @wraps(op)
    def __op__(self, other):
        if isinstance(other, dict):
            return type(self)({
                k: op(self[k], other[k]) 
                for k in self.keys() & other.keys()
            })
        elif isinstance(other, AbstractVariableOperator):
            return op(dict(self), other)
        else:
            return type(self)({k: op(val, other) for k, val in self.items()})
    
    return __op__

class VariableData(Dict[Variable, np.ndarray]):
    neg = _unary_op(operator.neg)
    abs = _unary_op(operator.abs)
    norm = _unary_op(np.linalg.norm)
    det = _unary_op(np.linalg.det)
        
    __add__ = _binary_op(operator.add)
    __radd__ = _binary_op(operator.add)
    __sub__ = _binary_op(operator.sub)
    __mul__ = _binary_op(operator.mul)
    __rmul__ = _binary_op(operator.mul)
    __truediv__ = _binary_op(operator.truediv)
    __pow__ = _binary_op(operator.pow)
    dot = _binary_op(np.dot)

    def map(self, func, *args, **kwargs):
        return type(self)(
            (k, func(val, *(arg[k] for arg in args), **kwargs))
            for k, val in self.items()
        )

    def subset(self, variables):
        return type(self)((v, self[v]) for v in variables)
    
    def __repr__(self):
        name = type(self).__name__
        data_repr = dict.__repr__(self)
        return f"{name}({data_repr})"
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.map(_apply_ufunc, *inputs, ufunc=ufunc, **kwargs)


def _apply_ufunc(val, *inputs, ufunc, **kwargs):
    return ufunc(val, *inputs, **kwargs)


class AbstractVariableOperator(ABC):
    """Implements the functionality of a linear operator acting
    on a dictionary of values indexed by `Variable` objects
    """
    @abstractmethod
    def __mul__(self, x: VariableData) -> VariableData:
        pass

    @abstractmethod
    def __rtruediv__(self, x: np.ndarray) -> VariableData:
        pass

    @abstractmethod
    def __rmul__(self, x: VariableData) -> VariableData:
        pass

    @abstractmethod
    def ldiv(self, x: VariableData) -> VariableData:
        pass

    def inv(self) -> 'InverseVariableOperator':
        return InverseVariableOperator(self)
        
    def quad(self, M: VariableData) -> VariableData:
        return (M * self).T * self

    def invquad(self, M: VariableData) -> VariableData:
        return (M / self).T / self



class InverseVariableOperator(AbstractVariableOperator):
    def __init__(self, operator):
        self.operator = operator 

    def __mul__(self, x: VariableData) -> VariableData:
        return self.operator.ldiv(x)

    def __rtruediv__(self, x: VariableData) -> VariableData:
        return x * self.operator

    def __rmul__(self, x: VariableData) -> VariableData:
        return x / self.operator

    def ldiv(self, x: VariableData) -> VariableData:
        return self * x

    def quad(self, M: VariableData) -> VariableData:
        return self.operator.invquad(M)

    def invquad(self, M: VariableData) -> VariableData:
        return self.operator.quad(M)

    def inv(self) -> AbstractVariableOperator:
        return self.operator
        
    @cached_property
    def log_det(self):
        return - self.operator.log_det


def _variable_binary_op(op):
    @wraps(op)
    def __op__(self, other):
        if isinstance(other, (dict, VariableData)):
            return VariableData({
                k: op(self.operators[k], other[k]) 
                for k in self.operators.keys() & other.keys()
            })
        elif isinstance(other, VariableOperator):
            return type(self)({
                v: op(self.operators[v], other.operators[v])
                for v in self.operators.keys() & other.operators.keys()
            })
        else:
            return type(self)({
                k: op(val, other) for k, val in self.operators.items()
            })
    
    return __op__

def ldiv(op, val):
    return op.ldiv(val)

class VariableOperator(AbstractVariableOperator):
    """
    """

    def __init__(self, operators: Dict[Variable, LinearOperator]):
        self.operators = operators

    __add__ = _variable_binary_op(operator.add)
    __sub__ = _variable_binary_op(operator.sub)
    __mul__ = _variable_binary_op(operator.mul)
    __rmul__ = _variable_binary_op(operator.mul)
    __rtruediv__ = _variable_binary_op(operator.truediv)
    ldiv = _variable_binary_op(ldiv)

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    def quad(self, values):
        return {
            v: H.T if np.ndim(H) else H for v, H in (values * self).items()
        } * self

    def invquad(self, values):
        return {
            v: H.T if np.ndim(H) else H for v, H in (values / self).items()
        } / self

    @cached_property
    def log_det(self):
        return sum(M.log_det for M in self.operators.values())

    @classmethod
    def from_scales(cls, scales):
        return cls({
            v: DiagonalMatrix(scale) for v, scale in scales.items()
        })
    
    @classmethod
    def from_covariances(cls, covs):
        return cls({
            v: CholeskyOperator.from_dense(cov).inv()
            for v, cov in covs.items()
        })

    @classmethod
    def from_pos_definite(cls, pos_defs):
        return cls({
            v: CholeskyOperator.from_dense(M) for v, M in pos_defs.items()
        })
        
    @classmethod
    def from_dense(cls, pos_defs):
        return cls({
            v: MatrixOperator(M) for v, M in pos_defs.items()
        })

    def diagonal(self):
        return type(self)({v: op.diagonal() for v, op in self.operators.items()})   

    def to_dense(self):
        return VariableData({v: op.to_dense() for v, op in self.operators.items()})


class CholeskyVariableOperator(AbstractVariableOperator):
    def __init__(self, cholesky, param_shapes):
        self.cholesky = cholesky
        self.param_shapes = param_shapes

    @classmethod 
    def from_dense(cls, M, param_shapes):
        return cls(
            CholeskyOperator.from_dense(M).inv(),
            param_shapes)

    @classmethod
    def from_optresult(cls, opt_result):
        param_shapes = opt_result.param_shapes

        cov = opt_result.result.hess_inv
        if not isinstance(cov, np.ndarray):
            # if optimiser is L-BFGS-B then convert
            # implicit hess_inv into dense matrix
            cov = cov.todense()

        return cls.from_dense(cov, param_shapes)

    def __mul__(self, values: VariableData) -> VariableData:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(M * x)

    def __rtruediv__(self, values: VariableData) -> VariableData:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(x / M)

    def __rmul__(self, values: VariableData) -> VariableData:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(x * M)

    @abstractmethod
    def ldiv(self, values: VariableData) -> VariableData:
        M, x = self.cholesky, self.param_shapes.flatten(values)
        return self.param_shapes.unflatten(M.ldiv(x))

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @cached_property
    def log_det(self):
        return self.cholesky.log_det


class IdentityVariableOperator(AbstractVariableOperator):
    def __init__(self):
        pass

    def _identity(self, values: VariableData) -> VariableData:
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


identity_operator = IdentityOperator()
identity_variable_operator = IdentityVariableOperator()