from itertools import chain, count
from typing import Optional, Tuple, Dict
import operator
from abc import ABC, abstractmethod
from functools import wraps, reduce
from math import sqrt

import numpy as np
from autoconf import cached_property

from autofit.mapper.model_object import ModelObject


class Plate:
    _ids = count()

    def __init__(self, name: Optional[str] = None):
        """
        Represents a dimension, such as number of observations, features or dimensions

        Parameters
        ----------
        name
            The name of this dimension
        """
        self.id = next(self._ids)
        self.name = name or f"plate_{self.id}"

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"

    def __eq__(self, other):
        return isinstance(other, Plate) and self.id == other.id

    def __hash__(self):
        return self.id

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id


class Variable(ModelObject):
    # __slots__ = ("name", "plates")

    def __init__(self, name: str = None, *plates: Plate, id_=None):
        """
        Represents a variable in the problem. This may be fixed data or some coefficient
        that we are optimising for.

        Parameters
        ----------
        name
            The name of this variable
        plates
            Representation of the dimensions of this variable
        """
        self.plates = plates
        super().__init__(id_=id_)
        self.name = name or f"{self.__class__.__name__.lower()}_{self.id}"

    def __repr__(self):
        args = ", ".join(chain([self.name], map(repr, self.plates)))
        return f"{self.__class__.__name__}({args})"

    def __hash__(self):
        return self.id

    def __len__(self):
        return len(self.plates)

    def __str__(self):
        return self.name

    @property
    def ndim(self) -> int:
        """
        How many dimensions does this variable have?
        """
        return len(self.plates)


def broadcast_plates(
    value: np.ndarray,
    in_plates: Tuple[Plate, ...],
    out_plates: Tuple[Plate, ...],
    reducer: np.ufunc = np.sum,
) -> np.ndarray:
    """
    Extract the indices of a collection of plates then match
    the shape of the data to that shape.

    Parameters
    ----------
    value
        A value to broadcast
    in_plates
        Plates representing the dimensions of the values
    out_plates
        Plates representing the output dimensions
    reducer
        function to reduce excess plates over, default np.sum
        must take axis as keyword argument


    Returns
    -------
    The value reshaped to match the plates
    """
    n_in = len(in_plates)
    n_out = len(out_plates)
    shift = np.ndim(value) - n_in
    if not (0 <= shift <= 1):
        raise ValueError("dimensions of value incompatible with passed plates")

    in_axes = list(range(shift, n_in + shift))
    out_axes = []
    k = n_out + shift

    for plate in in_plates:
        try:
            out_axes.append(out_plates.index(plate) + shift)
        except ValueError:
            out_axes.append(k)
            k += 1

    moved_value = np.moveaxis(
        np.expand_dims(value, tuple(range(n_in + shift, k))),
        in_axes,
        out_axes,
    )
    return reducer(moved_value, axis=tuple(range(n_out + shift, k)))


def _get_variable_data_class(data):
    # So that these methods work with standard python dictionaries
    return type(data) if isinstance(data, VariableData) else VariableData


def _unary_op(op):
    @wraps(op)
    def __op__(self):
        cls = _get_variable_data_class(self)
        return cls({k: op(val) for k, val in self.items()})

    return __op__


def _binary_op(op, ravel=False):
    if ravel:

        @wraps(op)
        def __op__(self, other):
            cls = _get_variable_data_class(self)
            if isinstance(other, dict):
                return cls(
                    {
                        k: op(np.ravel(self[k]), np.ravel(other[k]))
                        for k in self.keys() & other.keys()
                    }
                )
            elif isinstance(other, AbstractVariableOperator):
                return op(dict(self), other)
            else:
                return cls({k: op(val, other) for k, val in self.items()})

    else:

        @wraps(op)
        def __op__(self, other):
            cls = _get_variable_data_class(self)
            if isinstance(other, dict):
                return cls(
                    {k: op(self[k], other[k]) for k in self.keys() & other.keys()}
                )
            elif isinstance(other, AbstractVariableOperator):
                return op(dict(self), other)
            else:
                return cls({k: op(val, other) for k, val in self.items()})

    return __op__


class VariableData(Dict[Variable, np.ndarray]):
    neg = _unary_op(operator.neg)
    abs = _unary_op(operator.abs)
    var_norm = _unary_op(np.linalg.norm)
    var_det = _unary_op(np.linalg.det)
    var_max = _unary_op(np.max)
    var_min = _unary_op(np.min)
    var_sum = _unary_op(np.sum)
    var_prod = _unary_op(np.prod)
    var_isfinite = _unary_op(np.isfinite)

    __add__ = _binary_op(operator.add)
    __radd__ = _binary_op(operator.add)
    __sub__ = _binary_op(operator.sub)
    __mul__ = _binary_op(operator.mul)
    __rmul__ = _binary_op(operator.mul)
    __truediv__ = _binary_op(operator.truediv)
    __pow__ = _binary_op(operator.pow)

    sub = __sub__
    add = __add__
    mul = __mul__
    div = __truediv__
    var_dot = _binary_op(np.dot, ravel=True)

    @property
    def T(self):
        return type(self)({k: val.T for k, val in self.items()})

    @property
    def shapes(self):
        return {k: np.shape(val) for k, val in self.items()}

    def ravel(self):
        cls = _get_variable_data_class(self)
        return cls({k: np.ravel(val) for k, val in self.items()})

    def map(self, func, *args, **kwargs):
        cls = _get_variable_data_class(self)
        return cls(
            (k, func(val, *(arg[k] for arg in args), **kwargs))
            for k, val in self.items()
        )

    def reduce(self, func):
        return reduce(func, self.values())

    def mapreduce(self, func, op):
        return VariableData.map(self, func).reduce(op)

    def subset(self, variables):
        cls = _get_variable_data_class(self)
        return cls((v, self[v]) for v in variables)

    def sum(self) -> float:
        return sum(VariableData.var_sum(self).values())

    def prod(self) -> float:
        return VariableData.reduce(VariableData.var_prod(self).values(), operator.mul)

    def det(self) -> float:
        return VariableData.var_det(self).reduce(operator.mul)

    def log_det(self) -> float:
        return VariableData.mapreduce(self, np.linalg.logdet, operator.add)

    def max(self) -> float:
        return max(VariableData.var_max(self).values())

    def min(self) -> float:
        return min(VariableData.var_min(self).values())

    def dot(self, other) -> float:
        return VariableData.var_dot(self, other).sum()

    def norm(self) -> float:
        return sqrt(VariableData.dot(self, self))

    def vecnorm(self, ord: Optional[float] = None) -> float:
        if ord:
            absval = VariableData.abs(self)
            if ord == np.Inf:
                return absval.max()
            elif ord == -np.Inf:
                return absval.min()
            else:
                return (absval ** ord).sum() ** (1.0 / ord)
        else:
            return VariableData.norm(self)

    def __repr__(self):
        name = type(self).__name__
        data_repr = dict.__repr__(self)
        return f"{name}({data_repr})"


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

    def dot(self, x):
        return self * x

    __matmul__ = dot

    def inv(self) -> "InverseVariableOperator":
        return InverseVariableOperator(self)

    def quad(self, M: VariableData) -> VariableData:
        return (M * self).T * self

    def invquad(self, M: VariableData) -> VariableData:
        return (M / self).T / self

    @abstractmethod
    def update(self, *args: Tuple[VariableData, VariableData]):
        pass

    def lowrankupdate(self, *values: VariableData):
        return self.update(*((value, value) for value in values()))

    def lowrankdowndate(self, *values: VariableData):
        return self.update(*((value, VariableData.neg(value)) for value in values()))


class InverseVariableOperator(AbstractVariableOperator):
    def __init__(self, op):
        self.operator = op

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
        return -self.operator.log_det

    def update(self, *args: Tuple[VariableData, VariableData]):
        # apply Sherman-Morrison formulat
        A = self.operator
        for (u, v) in args:
            A1u = A * u
            A1v = v * A
            vTA1u = -A1u.dot(v)
            A = A.update(A1u, A1v * vTA1u)

        return type(self)(A)

    def to_full(self) -> "VariableFullOperator":
        full_op = self.operator.to_full()
        M = np.linalg.inv(full_op.operator.to_dense())
        return full_op.from_dense(M, full_op.param_shapes)

    def diagonal(self) -> VariableData:
        full_op = self.to_full()
        diag = full_op.operator.to_dense().diagonal()
        return full_op.param_shapes.unflatten(diag)

    def to_block(self) -> "VariableOperator":
        return self.to_full().to_block()
