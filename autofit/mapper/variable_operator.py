from functools import wraps
from typing import Dict, Tuple, Set, Sequence
import operator

import numpy as np
from scipy.linalg import block_diag

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
    VariableLinearOperator,
    InverseVariableOperator,
    rtruediv,
    rmul,
)

from autofit.graphical.utils import FlattenArrays


def ldiv(op, val):
    return op.ldiv(val)


def _merged_operator_vdata(op):
    @wraps(op)
    def __op__(self, *args):
        out = VariableData()
        for op in self.operators:
            variables = op.variables
            subsets = (VariableData.subset(arg, variables) for arg in args)
            out.update(op(*subsets))

        return out


class MergedVariableOperator(VariableLinearOperator):
    def __init__(self, *operators: Tuple[VariableLinearOperator]):
        self.operators = operators

    __mul__ = _merged_operator_vdata(operator.mul)
    __rmul__ = _merged_operator_vdata(operator.mul)
    __rtruediv__ = _merged_operator_vdata(operator.truediv)
    ldiv = _merged_operator_vdata(ldiv)

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @property
    def variables(self):
        variables = self.operators[0].variables
        for op in self.operators[1:]:
            variables = variables | op.variables

        return variables

    @property
    def is_diagonal(self):
        return all(op.is_diagonal for op in self.operators.values())

    def to_block(self):
        blocks = {}
        for op in self.operators:
            blocks.update(op.to_block().operators)

        return VariableOperator(blocks)

    def to_full(self):
        full_ops = [op.to_full() for op in self.operators]
        full = block_diag(*(op.operator.to_dense() for op in full_ops))
        param_shapes = FlattenArrays(
            (v, s) for op in full_ops for v, s in op.param_shapes.items()
        )
        return VariableFullOperator(full_ops[0].operator.from_dense(full), param_shapes)

    def inv(self):
        return type(self)(*(op.inv() for op in self.operators))

    @cached_property
    def log_det(self):
        return sum(M.log_det for M in self.operators.values())

    def diagonal(self):
        diag = VariableData()
        for op in self.operators:
            diag.update(op.diagonal())

        return diag

    def update(self, *args):
        operators = list(self.operators)
        for u, v in args:
            operators = [op.update(u, v) for op in operators]

        return MergedVariableOperator(operators)

    def lowrankupdate(self, *values: VariableData) -> "VariableFullOperator":
        operators = list(self.operators)
        for u in values:
            operators = [op.lowrankupdate(u) for op in operators]

        return MergedVariableOperator(operators)

    def lowrankdowndate(self, values: VariableData) -> "VariableFullOperator":
        operators = list(self.operators)
        for u in values:
            operators = [op.lowrankdowndate(u) for op in operators]

        return MergedVariableOperator(operators)

    def diagonalupdate(self, values: VariableData) -> "VariableFullOperator":
        operators = [op.diagonalupdate(values) for op in self.operators]
        return MergedVariableOperator(operators)


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
        elif isinstance(other, VariableLinearOperator):
            return op(self.to_full(), other.to_full())
        else:
            return type(self)({k: op(val, other) for k, val in self.operators.items()})

    return __op__


class VariableOperator(VariableLinearOperator):
    """ """

    def __init__(self, operators: Dict[Variable, LinearOperator]):
        self.operators = operators

    __add__ = _variable_binary_op(operator.add)
    __sub__ = _variable_binary_op(operator.sub)
    __mul__ = _variable_binary_op(operator.mul)
    __rmul__ = _variable_binary_op(rmul)
    __truediv__ = _variable_binary_op(operator.truediv)
    __rtruediv__ = _variable_binary_op(rtruediv)
    ldiv = _variable_binary_op(ldiv)

    rdiv = __rtruediv__
    rmul = __rmul__
    lmul = __mul__
    __matmul__ = __mul__

    @property
    def is_diagonal(self):
        return all(op.is_diagonal for op in self.operators.values())

    @property
    def variables(self) -> Set[Variable]:
        return self.operators.keys()

    def inv(self):
        return type(self)({v: op.inv() for v, op in self.operators.items()})

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

    @classmethod
    def from_diagonal(cls, diag: VariableData) -> "VariableFullOperator":
        operators = {v: DiagonalMatrix(d) for v, d in diag.items()}
        return cls(operators)

    def diagonal(self):
        return VariableData({v: op.diagonal() for v, op in self.operators.items()})

    def to_diagonal(self) -> "VariableOperator":
        return type(self)({v: DiagonalMatrix(d) for v, d in self.diagonal().items()})

    def blocks(self) -> VariableData:
        return VariableData({v: op.to_dense() for v, op in self.operators.items()})

    def to_block(self) -> "VariableOperator":
        return self

    @cached_property
    def param_shapes(self):
        return FlattenArrays({v: op.lshape for v, op in self.operators.items()})

    def to_full(self) -> "VariableFullOperator":
        M = self.param_shapes.flatten2d(
            {v: op.to_dense() for v, op in self.operators.items()}
        )
        return VariableFullOperator(MatrixOperator(M), self.param_shapes)

    def update(self, *args: Tuple[VariableData, VariableData]):
        operators = self.operators.copy()
        for (u, v) in args:
            for k in operators.keys() & u.keys() & v.keys():
                operators[k] = operators[k].update(u[k], v[k])

        return type(self)(operators)

    def diagonalupdate(self, d: VariableData):
        return type(self)(
            {
                k: self.operators[k].diagonalupdate(d[k])
                for k in self.operators.keys() & d.keys()
            }
        )


def _variablefull_binary_op(op):
    @wraps(op)
    def __op__(self: "VariableFullOperator", other):
        if isinstance(other, dict):
            return self.param_shapes.unflatten(
                op(self.operator, self.param_shapes.flatten(other))
            )
        if isinstance(other, VariableFullOperator):
            other = other.operator
        elif isinstance(other, VariableLinearOperator):
            other = other.to_full().operator

        op_new = op(self.operator, other)
        return type(self)(op_new, self.param_shapes)

    return __op__


class VariableFullOperator(VariableLinearOperator):

    __add__ = _variablefull_binary_op(operator.add)
    __sub__ = _variablefull_binary_op(operator.sub)
    __mul__ = _variablefull_binary_op(operator.mul)
    __rmul__ = _variablefull_binary_op(rmul)
    __truediv__ = _variablefull_binary_op(operator.truediv)
    __rtruediv__ = _variablefull_binary_op(rtruediv)
    ldiv = _variablefull_binary_op(ldiv)
    rdiv = __rtruediv__
    rmul = __rmul__
    mul = __mul__
    __matmul__ = __mul__

    def __init__(self, op: LinearOperator, param_shapes: FlattenArrays):
        self.operator = op
        self.param_shapes = param_shapes

    @property
    def is_diagonal(self):
        return self.operator.is_diagonal

    @property
    def variables(self) -> Set[Variable]:
        return self.param_shapes.keys()

    @classmethod
    def from_posdef(
        cls, M: np.ndarray, param_shapes: FlattenArrays
    ) -> "VariableFullOperator":
        return cls(CholeskyOperator.from_dense(M), param_shapes)

    @classmethod
    def from_dense(
        cls, M: np.ndarray, param_shapes: FlattenArrays
    ) -> "VariableFullOperator":
        return cls(MatrixOperator.from_dense(M), param_shapes)

    @classmethod
    def from_diagonal(cls, diag: VariableData) -> "VariableFullOperator":
        param_shapes = FlattenArrays.from_arrays(diag)
        operator = DiagonalMatrix(param_shapes.flatten(diag))
        return cls(operator, param_shapes)

    @classmethod
    def from_blocks(cls, Ms: VariableData) -> "VariableFullOperator":
        param_shapes = FlattenArrays(
            {v: np.shape(M)[: np.ndim(M) // 2] for v, M in Ms.items()}
        )
        return cls.from_dense(param_shapes.flatten2d(Ms), param_shapes)

    def blocks(self) -> VariableData:
        return self.param_shapes.unflatten2d(self.operator.to_dense())

    def to_block(self, cls=None) -> VariableOperator:
        blocks = self.blocks()
        cls = cls or type(self.operator)
        return VariableOperator({k: cls.from_dense(M) for k, M in blocks.items()})

    def diagonal(self) -> VariableData:
        return self.param_shapes.unflatten(self.operator.diagonal())

    def to_diagonal(self):
        diagonal = DiagonalMatrix(self.operator.diagonal())
        return type(self)(diagonal, self.param_shapes)

    def to_full(self):
        return self

    @classmethod
    def from_optresult(cls, opt_result) -> "VariableFullOperator":
        param_shapes = opt_result.param_shapes

        cov = opt_result.result.hess_inv
        if not isinstance(cov, np.ndarray):
            # if optimiser is L-BFGS-B then convert
            # implicit hess_inv into dense matrix
            cov = cov.todense()

        return cls.from_dense(cov, param_shapes)

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

    def diagonalupdate(self, values: VariableData) -> "VariableFullOperator":
        v = self.param_shapes.flatten(values)
        return type(self)(self.operator.diagonalupdate(v), self.param_shapes)


class IdentityVariableOperator(VariableLinearOperator):
    def __init__(self, variables=()):
        self._variables = set(variables)

    def _identity(self, values: VariableData) -> VariableData:
        return values

    @property
    def variables(self):
        return self._variables

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

    def update(self, *args):
        (u, v), *next_args = args
        param_shapes = FlattenArrays.from_arrays(u)
        uv = param_shapes.flatten(u)[:, None] * param_shapes.flatten(v)[None, :]
        out = VariableFullOperator.from_dense(uv, param_shapes)
        return out.update(*next_args)


identity_operator = IdentityOperator()
identity_variable_operator = IdentityVariableOperator()
