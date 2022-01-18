from functools import wraps
from typing import Dict
import operator

import numpy as np

from autoconf import cached_property

from autofit.mapper.operator import (
    LinearOperator,
    IdentityOperator,
    DiagonalMatrix,
    CholeskyOperator,
    MatrixOperator,
    QROperator,
)
from autofit.mapper.variable import (
    Variable,
    VariableData,
    AbstractVariableOperator,
    InverseVariableOperator,
)

from autofit.graphical.utils import FlattenArrays


def ldiv(op, val):
    return op.ldiv(val)


def _variable_binary_op(op):
    @wraps(op)
    def __op__(self, other):
        if isinstance(other, (dict, VariableData)):
            return VariableData(
                {
                    k: op(self.operators[k], other[k])
                    for k in self.operators.keys() & other.keys()
                }
            )
        elif isinstance(other, VariableOperator):
            return type(self)(
                {
                    v: op(self.operators[v], other.operators[v])
                    for v in self.operators.keys() & other.operators.keys()
                }
            )
        else:
            return type(self)({k: op(val, other) for k, val in self.operators.items()})

    return __op__


class VariableOperator(AbstractVariableOperator):
    """ """

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

    @cached_property
    def log_det(self):
        return sum(M.log_det for M in self.operators.values())

    @classmethod
    def from_scales(cls, scales):
        return cls({v: DiagonalMatrix(scale) for v, scale in scales.items()})

    @classmethod
    def from_covariances(cls, covs):
        return cls(
            {v: CholeskyOperator.from_dense(cov).inv() for v, cov in covs.items()}
        )

    @classmethod
    def from_pos_definite(cls, pos_defs):
        return cls({v: CholeskyOperator.from_dense(M) for v, M in pos_defs.items()})

    @classmethod
    def from_dense(cls, pos_defs):
        return cls({v: MatrixOperator(M) for v, M in pos_defs.items()})

    def diagonal(self):
        return VariableData({v: op.diagonal() for v, op in self.operators.items()})

    def to_dense(self):
        return VariableData({v: op.to_dense() for v, op in self.operators.items()})


def _variablefull_binary_op(op):
    @wraps(op)
    def __op__(self: "VariableFullOperator", other):
        if isinstance(other, (dict, VariableData)):
            return self.param_shapes.unflatten(
                op(self.operator, self.param_shapes.flatten(other))
            )

        if isinstance(other, VariableFullOperator):
            other = other.operator

        op_new = op(self.operator, other)
        return type(self)(op_new, self.param_shapes)

    return __op__


class VariableFullOperator(AbstractVariableOperator):
    def __init__(self, op: LinearOperator, param_shapes: FlattenArrays):
        self.operator = op
        self.param_shapes = param_shapes

    @classmethod
    def from_posdef(
        cls, M: np.ndarray, param_shapes: FlattenArrays
    ) -> "VariableFullOperator":
        return cls(CholeskyOperator.from_dense(M), param_shapes)

    @classmethod
    def from_dense(
        cls, M: np.ndarray, param_shapes: FlattenArrays
    ) -> "VariableFullOperator":
        return cls(QROperator.from_dense(M), param_shapes)

    def to_block(self, cls=None) -> VariableOperator:
        blocks = self.param_shapes.unflatten2d(self.operator.to_dense())
        cls = cls or type(self.operator)
        return VariableOperator({k: cls.from_dense(M) for k, M in blocks.items()})

    def diagonal(self) -> VariableData:
        return self.to_block(MatrixOperator).diagonal()

    @classmethod
    def from_optresult(cls, opt_result) -> "VariableFullOperator":
        param_shapes = opt_result.param_shapes

        cov = opt_result.result.hess_inv
        if not isinstance(cov, np.ndarray):
            # if optimiser is L-BFGS-B then convert
            # implicit hess_inv into dense matrix
            cov = cov.todense()

        return cls.from_dense(cov, param_shapes)

    __add__ = _variablefull_binary_op(operator.add)
    __sub__ = _variablefull_binary_op(operator.sub)
    __mul__ = _variablefull_binary_op(operator.mul)
    __rmul__ = _variablefull_binary_op(operator.mul)
    __rtruediv__ = _variablefull_binary_op(operator.truediv)
    ldiv = _variablefull_binary_op(ldiv)
    rdiv = __rtruediv__
    rmul = __rmul__
    mul = __mul__
    __matmul__ = __mul__

    @cached_property
    def log_det(self):
        return self.operator.log_det

    def update(self, *args):
        op = self.operator.update(
            *(
                (self.param_shapes.flatten(u), self.param_shapes.flatten(v))
                for u, v in args
            )
        )
        return type(self)(op, self.param_shapes)

    def lowrankupdate(self, values: VariableData) -> "VariableFullOperator":
        v = self.param_shapes.flatten(values)
        return type(self)(self.operator.lowrankupdate(v), self.param_shapes)

    def lowrankdowndate(self, values: VariableData) -> "VariableFullOperator":
        v = self.param_shapes.flatten(values)
        return type(self)(self.operator.lowrankdowndate(v), self.param_shapes)


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
        return 0.0


identity_operator = IdentityOperator()
identity_variable_operator = IdentityVariableOperator()
