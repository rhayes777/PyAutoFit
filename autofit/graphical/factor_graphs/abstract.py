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
from autofit.graphical.factor_graphs.jacobians import (
    AbstractJacobian,
    JacobianVectorProduct,
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

    @property
    def _unique_representation(self):
        return (
            self._factor,
            self.arg_names,
            self.args,
            frozenset(self._deterministic_variables),
        )

    def __hash__(self):
        return hash(self._unique_representation)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._unique_representation == other._unique_representation
        return False

    def _factor_args(self, *args):
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
        args = tuple(np.array(value, dtype=np.float64) for value in args)

        raw_fval0 = self._factor_args(*args)
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
                    fval1 = self._factor_value(self._factor_args(*args)).to_dict()
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

    def jacobian(
        self, values: Dict[Variable, np.array], eps=1e-6
    ) -> "AbstractJacobian":
        return self.func_jacobian(values, eps=eps)[1]

    def func_gradient(self, values: VariableData) -> Tuple[FactorValue, GradientValue]:
        fval, fjac = self.func_jacobian(values)
        return fval, fjac.grad()

    def flatten(self, param_shapes: FlattenArrays) -> "FlattenedNode":
        return FlattenedNode(self, param_shapes)


class AbstractFactor(AbstractNode, ABC):
    def __init__(
        self,
        name="",
        plates: Tuple[Plate, ...] = (),
        **kwargs: Variable,
    ):
        super().__init__(plates=plates, **kwargs)
        self._name = name or f"factor_{self.id}"
        self._deterministic_variables = set()

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name

    @property
    def deterministic_variables(self) -> Tuple[Variable]:
        return self._deterministic_variables

    @property
    def name(self) -> str:
        return self._name

    def __mul__(self, other):
        """
        When two factors are multiplied together this creates a graph
        """
        from autofit.graphical.factor_graphs.graph import FactorGraph

        return FactorGraph([self]) * other

    @property
    def variables(self) -> Set[Variable]:
        """
        Dictionary mapping the names of variables to those variables
        """
        return set(self._kwargs.values())

    @property
    def _kwargs_dims(self) -> Dict[str, int]:
        """
        The number of plates for each keyword argument variable
        """
        return {key: len(value) for key, value in self._kwargs.items()}

    @cached_property
    def _variable_plates(self) -> Dict[str, np.ndarray]:
        """
        Maps the name of each variable to the indices of its plates
        within this node
        """
        return {
            variable: self._match_plates(variable.plates)
            for variable in self.all_variables
        }

    @property
    def n_deterministic(self) -> int:
        """
        How many deterministic variables are there associated with this node?
        """
        return len(self._deterministic_variables)

    def _resolve_args(self, **kwargs: np.ndarray) -> dict:
        """
        Transforms in the input arguments to match the arguments
        specified for the factor.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        return {n: kwargs[v.name] for n, v in self._kwargs.items()}

    def _set_factor(self, factor):
        self._factor = factor
        self._has_exact_projection = getattr(factor, "has_exact_projection", None)
        self._calc_exact_projection = getattr(factor, "calc_exact_projection", None)
        self._calc_exact_update = getattr(factor, "calc_exact_update", None)

    def has_exact_projection(self, mean_field) -> bool:
        if self._has_exact_projection:
            return self._has_exact_projection(**self.resolve_variable_dict(mean_field))
        else:
            return False

    def calc_exact_projection(self, mean_field) -> "MeanField":
        if self._calc_exact_projection:
            from autofit.graphical.mean_field import MeanField

            projection = self._calc_exact_projection(
                **self.resolve_variable_dict(mean_field)
            )
            return MeanField({self._kwargs[v]: dist for v, dist in projection.items()})
        else:
            return NotImplementedError

    def calc_exact_update(self, mean_field) -> "MeanField":
        if self._calc_exact_update:
            from autofit.graphical.mean_field import MeanField

            projection = self._calc_exact_update(
                **self.resolve_variable_dict(mean_field)
            )
            return MeanField({self._kwargs[v]: dist for v, dist in projection.items()})
        else:
            return NotImplementedError

    def safe_exact_update(self, mean_field) -> Tuple[bool, "MeanField"]:
        if self._has_exact_projection:
            from autofit.graphical.mean_field import MeanField

            _mean_field = self.resolve_variable_dict(mean_field)
            if self._has_exact_projection(**_mean_field):
                projection = self._calc_exact_update(**_mean_field)
                return True, MeanField(
                    {self._kwargs[v]: dist for v, dist in projection.items()}
                )

        return False, mean_field

    def name_for_variable(self, variable):
        return self.name


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


from autofit.graphical.mean_field import MeanField
