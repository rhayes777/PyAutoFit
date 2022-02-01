from abc import ABC, abstractmethod
from itertools import count
from typing import (
    List,
    Tuple,
    Dict,
    cast,
    Set,
    Optional,
    Union,
    Collection,
    Any,
    Callable,
)

# from autofit.graphical.factor_graphs.factor import Factor

Protocol = ABC  # for python 3.7 compat

import numpy as np

from autoconf import cached_property
from autofit.graphical.utils import (
    FlattenArrays,
    Axis,
    nested_filter,
    nested_update,
    is_variable,
)
from autofit.mapper.variable import (
    Variable,
    Plate,
    FactorValue,
    VariableData,
    variables,
    VariableLinearOperator,
)

Value = Dict[Variable, np.ndarray]


GradientValue = VariableData
HessianValue = Any


class FactorInterface(Protocol):
    def __call__(self, values: Value) -> FactorValue:
        pass


class FactorGradientInterface(Protocol):
    def __call__(self, values: Value) -> Tuple[FactorValue, GradientValue]:
        pass


from autofit.graphical.factor_graphs.numerical import (
    # numerical_func_jacobian,
    numerical_func_jacobian_hessian,
)

from autofit.mapper.variable_operator import (
    RectVariableOperator,
    LinearOperator,
    VariableOperator,
)


class AbstractJacobian(VariableLinearOperator):
    """
    Examples
    --------
    def linear(x, a, b):
        z = x.dot(a) + b
        return (z**2).sum(), z

    def full(x, a, b):
        z2, z = linear(x, a, b)
        return z2 + z.sum()

    x_, a_, b_, y_, z_ = variables("x, a, b, y, z")
    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    # values = {x_: x, y_: y, a_: a, b_: b}

    linear_factor_jvp = FactorJVP(
        linear, x_, a_, b_, factor_out=(FactorValue, z_))

    linear_factor_vjp = FactorVJP(
        linear, x_, a_, b_, factor_out=(FactorValue, z_))

    values = {x_: x, a_: a, b_: b}

    jvp_val, jvp_jac = linear_factor_jvp.func_jacobian(values)
    vjp_val, vjp_jac = linear_factor_vjp.func_jacobian(values)


    assert np.allclose(vjp_val, jvp_val)
    assert (vjp_jac(vjp_val) - jvp_jac(vjp_val)).norm() == 0
    """

    def __call__(self, values):
        return self.__rmul__(values)

    def __str__(self) -> str:
        out_var = str(
            nested_update(self.factor_out, {v: v.name for v in self.out_variables})
        ).replace("'", "")

        in_var = ", ".join(v.name for v in self.variables)
        cls_name = type(self).__name__
        return f"{cls_name}({out_var} → ∂({in_var})ᵀ {out_var})"

    __repr__ = __str__

    def _full_repr(self) -> str:
        out_var = str(self.factor_out)
        in_var = str(self.variables)
        cls_name = type(self).__name__
        return f"{cls_name}({out_var} → ∂({in_var})ᵀ {out_var})"

    def grad(self, values=None):
        grad = VariableData({FactorValue: 1.0})
        if values:
            grad.update(values)

        for v, g in self(grad).items():
            grad[v] = grad.get(v, 0) + g

        return grad


class JacobianVectorProduct(AbstractJacobian, RectVariableOperator):
    __init__ = RectVariableOperator.__init__

    @property
    def variables(self):
        return self.left_variables

    @property
    def out_variables(self):
        return self.right_variables

    @property
    def factor_out(self):
        return tuple(self.out_variables)


class VectorJacobianProduct(AbstractJacobian):
    def __init__(
        self, factor_out, vjp: Callable, *variables: Variable, out_shapes=None
    ):
        self.factor_out = factor_out
        self.vjp = vjp
        self._variables = variables
        self.out_shapes = out_shapes

    @property
    def variables(self):
        return self._variables

    @cached_property
    def out_variables(self):
        return set(v[0] for v in nested_filter(is_variable, self.factor_out))

    def _get_cotangent(self, values):
        if isinstance(values, FactorValue):
            values = values.to_dict()

        if isinstance(values, dict):
            if self.out_shapes:
                for v in self.out_shapes.keys() - values.keys():
                    values[v] = np.zeros(self.out_shapes[v])
            out = nested_update(self.factor_out, values)
            return out

        if isinstance(values, int):
            values = float(values)

        return values

    def __call__(self, values: Union[VariableData, FactorValue]) -> VariableData:
        v = self._get_cotangent(values)
        grads = self.vjp(v)
        return VariableData(zip(self.variables, grads))

    __rmul__ = __call__

    def _not_implemented(self, *args):
        raise NotImplementedError()

    __rtruediv__ = _not_implemented
    ldiv = _not_implemented
    __mul__ = _not_implemented
    update = _not_implemented


class AbstractNode(ABC):
    _deterministic_variables: Tuple[Variable, ...] = ()
    _plates: Tuple[Plate, ...] = ()
    _factor: callable = None
    _id = count()
    factor_out = FactorValue
    eps = 1e-6

    def __init__(self, plates: Tuple[Variable, ...] = (), **kwargs: Variable):
        """
        A node in a factor graph

        Parameters
        ----------
        args
            Positional arguments passed to the factor
        kwargs
            Key word arguments passed to the value
        """
        self._plates = plates
        self._kwargs = kwargs
        self._variable_name_kw = {v.name: kw for kw, v in kwargs.items()}
        self.id = next(self._id)

    def resolve_variable_dict(
        self, variable_dict: Dict[Variable, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return {
            self._variable_name_kw[v.name]: x
            for v, x in variable_dict.items()
            if v.name in self._variable_name_kw
        }

    @property
    @abstractmethod
    def variables(self):
        pass

    @property
    def free_variables(self):
        return self.variables - getattr(self, "fixed_values", {}).keys()

    @property
    def args(self):
        return tuple(self._kwargs.values())

    @property
    def arg_names(self):
        return tuple(self._kwargs)

    @property
    def n_args(self):
        return len(self.args)

    @property
    def name_variable_dict(self) -> Dict[str, Variable]:
        return {variable.name: variable for variable in self.variables}

    @property
    @abstractmethod
    def deterministic_variables(self):
        ...

    @property
    def plates(self):
        return self._plates

    @cached_property
    def sorted_plates(self) -> Tuple[Plate]:
        """
        A tuple of the set of all plates in this graph, ordered by id
        """
        return tuple(
            sorted(
                set(
                    cast(Plate, plate)
                    for variable in self.all_variables
                    for plate in variable.plates
                )
            )
        )

    def __getitem__(self, item):
        try:
            return self._kwargs[item]
        except KeyError as e:
            for variable in self.variables | self._deterministic_variables:
                if variable.name == item:
                    return variable
            raise AttributeError(f"No attribute {item}") from e

    @property
    @abstractmethod
    def name(self) -> str:
        """
        A name for this object
        """

    @property
    def call_signature(self) -> str:
        """
        The apparent signature of this object
        """
        call_str = ", ".join(map("{0}={0}".format, self.kwarg_names))
        call_sig = f"{self.name}({call_str})"
        return call_sig

    @property
    def kwarg_names(self) -> List[str]:
        """
        The names of the variables passed as keyword arguments
        """
        return [arg.name for arg in self._kwargs.values()]

    @property
    def all_variables(self) -> Set[Variable]:
        """
        A dictionary of variables associated with this node
        """
        return self.variables | self.deterministic_variables

    def broadcast_plates(self, plates: Tuple[Plate], value: np.ndarray) -> np.ndarray:
        """
        Extract the indices of a collection of plates then match
        the shape of the data to that shape.

        Parameters
        ----------
        plates
            Plates representing the dimensions of some factor
        value
            A value to broadcast

        Returns
        -------
        The value reshaped to match the plates
        """
        shift = np.ndim(value) - len(plates)
        if shift > 1 or shift < 0:
            raise ValueError("dimensions of value incompatible with passed plates")
        reduce_axes = tuple(
            i + shift for i, p in enumerate(plates) if p and p not in self.plates
        )
        source_axes = [i + shift for i, p in enumerate(plates) if p in self.plates]
        destination_axes = [
            self.plates.index(plates[i - shift]) + shift for i in source_axes
        ]
        return np.moveaxis(
            np.sum(value, axis=reduce_axes), source_axes, destination_axes
        )

    def _broadcast(self, plate_inds: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Ensure the shape of the data matches the shape of the plates

        Parameters
        ----------
        plate_inds
            The indices of the plates of some factor within this node
        value
            Some data

        Returns
        -------
        The data reshaped
        """
        shape = np.shape(value)
        shift = len(shape) - plate_inds.size

        assert shift in {0, 1}, shift
        newshape = np.ones(self.ndim + shift, dtype=int)
        newshape[:shift] = shape[:shift]
        newshape[shift + plate_inds] = shape[shift:]

        # reorder axes of value to match ordering of newshape
        movedvalue = np.moveaxis(
            value, np.arange(plate_inds.size) + shift, np.argsort(plate_inds) + shift
        )
        return np.reshape(movedvalue, newshape)

    def _broadcast2d(self, plate_inds: np.ndarray, value: np.ndarray) -> np.ndarray:
        """
        Ensure the shape of the data matches the shape of the plates

        Parameters
        ----------
        plate_inds
            The indices of the plates of some factor within this node
        value
            Some data

        Returns
        -------
        The data reshaped
        """
        shape2d = np.shape(value)
        ndim = len(shape2d) // 2
        shape1, shape2 = shape2d[:ndim], shape2d[ndim:]

        newshape = np.ones(self.ndim * 2)
        newshape[plate_inds] = shape1
        newshape[plate_inds + self.ndim] = shape2

        # reorder axes of value to match ordering of newshape
        plate_order = np.argsort(plate_inds)
        movedvalue = np.moveaxis(
            value,
            np.arange(plate_inds.size * 2),
            np.r_[plate_order, plate_order + ndim],
        )
        return np.reshape(movedvalue, newshape)

    @property
    def ndim(self) -> int:
        """
        The number of plates contained within this graph's variables

        That is, the total dimensions of those variables.
        """
        return len(self.plates)

    def _match_plates(self, plates: Collection[Plate]) -> np.ndarray:
        """
        Find indices plates from some factor in the collection of
        plates associated with this node.

        Parameters
        ----------
        plates
            Plates from some other node

        Returns
        -------
        An array of plate indices
        """
        return np.array([self.plates.index(p) for p in plates], dtype=int)

    @abstractmethod
    def __call__(self, **kwargs) -> FactorValue:
        pass

    def __hash__(self):
        return hash(
            (
                self._factor,
                frozenset(self.name_variable_dict.items()),
                frozenset(self._deterministic_variables),
            )
        )

    def _factor_value(self, raw_fval):
        """Converts the raw output of the factor into a `FactorValue`
        where the values of the deterministic values are stored in a dict
        attribute `FactorValue.deterministic_values`
        """
        det_values = VariableData(nested_filter(is_variable, self.factor_out, raw_fval))
        fval = det_values.pop(FactorValue, 0.0)
        return FactorValue(fval, det_values)

    def _unpack_jacobian_out(self, raw_jac: Any) -> Dict[Variable, VariableData]:
        jac = {}
        for v0, vjac in nested_filter(is_variable, self.factor_out, raw_jac):
            jac[v0] = VariableData()
            for v1, j in zip(self.args, vjac):
                jac[v0][v1] = j

        return jac

    def _jac_out_to_jvp(
        self, raw_jac: Any, values: VariableData
    ) -> JacobianVectorProduct:
        jac = self._unpack_jacobian_out(raw_jac)
        return JacobianVectorProduct.from_dense(jac, values=values)

    def _call_args(self, *args):
        return self._factor(**dict(zip(self.arg_names, args)))

    def _numerical_factor_jacobian(
        self, *args, eps: Optional[float] = None
    ) -> Tuple[Any, Any]:
        """Calculates the dense numerical jacobian matrix with respect to
        the input arguments, broadly speaking, the following should return the
        same values (within numerical precision of the finite differences)

        factor._numerical_factor_jacobian(*args)

        factor._factor(*args), jax.jacobian(factor._factor, range(len(args)))(*args)
        """
        eps = eps or self.eps
        args = tuple(np.array(value) for value in args)

        raw_fval0 = self._call_args(*args)
        fval0 = self._factor_value(raw_fval0).to_dict()

        jac = {
            v0: tuple(
                np.empty_like(val, shape=np.shape(val) + np.shape(value))
                for value in args
            )
            for v0, val in fval0.items()
        }
        for i, val in enumerate(args):
            with np.nditer(val, op_flags=["readwrite"], flags=["multi_index"]) as it:
                for x_i in it:
                    val[it.multi_index] += eps
                    fval1 = self._factor_value(self._call_args(*args)).to_dict()
                    jac_v1_i = (fval1 - fval0) / eps
                    x_i -= eps
                    indexes = (Ellipsis,) + it.multi_index
                    for v0, jac_v0v_i in jac_v1_i.items():
                        jac[v0][i][indexes] = jac_v0v_i

        # This replicates the output of normal
        # jax.jacobian(self.factor, len(self.args))(*args)
        jac_out = nested_update(self.factor_out, jac)

        return raw_fval0, jac_out

    def numerical_func_jacobian(
        self, values: VariableData, **kwargs
    ) -> Tuple[FactorValue, JacobianVectorProduct]:
        args = (values[k] for k in self.args)
        raw_fval, raw_jac = self._numerical_factor_jacobian(*args, **kwargs)
        fval = self._factor_value(raw_fval)
        jvp = self._jac_out_to_jvp(raw_jac, values=fval.to_dict().merge(values))
        return fval, jvp

    func_jacobian = numerical_func_jacobian

    def jacobian(
        self,
        values: Dict[Variable, np.array],
        variables: Optional[Tuple[Variable, ...]] = None,
        _eps: float = 1e-6,
        _calc_deterministic: bool = True,
    ) -> "AbstractJacobian":
        return self.func_jacobian(
            values, variables, _eps=_eps, _calc_deterministic=_calc_deterministic
        )[1]

    def hessian(
        self,
        values: Dict[Variable, np.array],
        variables: Optional[Tuple[Variable, ...]] = None,
        _eps: float = 1e-6,
        _calc_deterministic: bool = True,
    ) -> "AbstractHessian":
        return self.func_jacobian_hessian(
            values, variables, _eps=_eps, _calc_deterministic=_calc_deterministic
        )[2]

    def func_gradient(self, values: VariableData) -> Tuple[FactorValue, GradientValue]:
        fval, fjac = self.func_jacobian(values)
        return fval, fjac.grad()

    def flatten(self, param_shapes: FlattenArrays) -> "FlattenedNode":
        return FlattenedNode(self, param_shapes)


class FlattenedNode:
    def __init__(self, node: "AbstractNode", param_shapes: FlattenArrays):
        self.node = node
        self.param_shapes = param_shapes

    def flatten(self, values: Value) -> np.ndarray:
        return self.param_shapes.flatten(values)

    def unflatten(self, x0: np.ndarray) -> Value:
        return self.param_shapes.unflatten(x0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = self.unflatten(x)
        return self.node(values)

    def func_jacobian(self, x: np.ndarray):
        values = self.unflatten(x)
        fval, jval = self.node.func_jacobian(values)
        grad = self.flatten(jval)
        return fval, grad

    def func_jacobian_hessian(self, x: np.ndarray):
        values = self.unflatten(x)
        fval, jval, hval = self.node.func_jacobian_hessian(values)
        grad = self.flatten(jval)
        hess = self.param_shapes.flatten2d(hval)
        return fval, grad, hess

    def jacobian(self, x: np.ndarray):
        return self.func_jacobian(x)[1]

    def hessian(self, x: np.ndarray):
        return self.func_jacobian_hessian(x)[1]

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.node, name)
