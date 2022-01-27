from itertools import repeat, chain
from typing import Tuple, Dict, Callable, Optional, Union
from inspect import getfullargspec

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier

import jax

from autoconf import cached_property

from autofit.graphical.factor_graphs.abstract import FactorValue, JacobianValue
from autofit.graphical.factor_graphs.factor import (
    AbstractFactor,
    Factor,
    DeterministicFactor,
)
from autofit.graphical.utils import aggregate, Axis, nested_filter, nested_update
from autofit.mapper.variable import (
    Variable,
    Plate,
    VariableLinearOperator,
    VariableData,
)
from autofit.mapper.variable_operator import (
    RectVariableOperator,
    LinearOperator,
    VariableOperator,
)


def _is_variable(v, *args):
    return isinstance(v, Variable)


class AbstractJacobian(VariableLinearOperator):
    def __call__(self, values):
        return self * values

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
        self, factor_out, vjp: Callable, *variables: Variable, fill_zero=False
    ):
        self.factor_out = factor_out
        self.vjp = vjp
        self._variables = variables
        self.fill_zero = fill_zero

    @property
    def variables(self):
        return self._variables

    @cached_property
    def out_variables(self):
        return set(v[0] for v in nested_filter(_is_variable, self.factor_out))

    def _get_cotangent(self, value):
        if isinstance(value, FactorValue):
            value = value.to_dict()

        if isinstance(value, dict):
            if self.fill_zero:
                for v in self.out_variables:
                    value.setdefault(v, 0.0)
            return nested_update(self.factor_out, value)

        if isinstance(value, int):
            value = float(value)

        return value

    def __mul__(self, value: Union[VariableData, FactorValue]) -> VariableData:
        v = self._get_cotangent(value)
        grads = self.vjp(v)
        return VariableData(zip(self.variables, grads))

    def _not_implemented(self, *args):
        raise NotImplementedError()

    __rtruediv__ = _not_implemented
    ldiv = _not_implemented
    __rmul__ = _not_implemented
    update = _not_implemented


class FactorVJP(Factor):
    def __init__(
        self, factor, *args: Variable, name="", factor_out=FactorValue, factor_vjp=None
    ):
        self._set_factor(factor)
        if factor_vjp:
            self._factor_vjp = factor_vjp

        self.args = args
        self.arg_names = [arg for arg in getfullargspec(factor).args]
        self.factor_out = factor_out

        AbstractFactor.__init__(
            self, **dict(zip(self.arg_names, self.args)), name=name or factor.__name__
        )

        det_variables = set(v[0] for v in nested_filter(_is_variable, factor_out))
        det_variables.discard(FactorValue)
        self._deterministic_variables = det_variables

    def _factor_value(self, raw_fval):
        det_values = VariableData(
            nested_filter(_is_variable, self.factor_out, raw_fval)
        )
        fval = det_values.pop(FactorValue, 0.0)
        return FactorValue(fval, det_values)

    def __call__(self, values: VariableData, axis=None):
        raw_fval = self._factor(*(values[v] for v in self.args))
        return self._factor_value(raw_fval)

    def _factor_vjp(self, *args):
        return jax.vjp(self._factor, *args)

    def func_jacobian(self, values: VariableData, axis=None):
        raw_fval, fvjp = self._factor_vjp(*(values[v] for v in self.args))
        fval = self._factor_value(raw_fval)
        fvjp_op = VectorJacobianProduct(self.factor_out, fvjp, *self.args)
        return fval, fvjp_op


class FactorJVP(FactorVJP):
    def __init__(
        self,
        factor,
        *args: Variable,
        name="",
        factor_out=FactorValue,
        func_jacobian=None,
    ):
        FactorVJP.__init__(self, factor, *args, name=name, factor_out=factor_out)
        self._jacobian = func_jacobian or jax.jacfwd(factor, list(range(len(args))))

    def func_jacobian(self, values: VariableData, axis=None):
        fval = self(values, axis=axis)
        raw_jac = self._jacobian(*(values[v] for v in self.args))

        jac = {v1: {} for v1 in self.args}
        for v0, vjac in nested_filter(_is_variable, self.factor_out, raw_jac):
            for v1, j in zip(self.args, vjac):
                jac[v1][v0] = j

        jvp = JacobianVectorProduct.from_dense(jac, values=fval.to_dict().merge(values))

        return fval, jvp


#################################################################


class FactorJacobian(Factor):
    """
    A node in a graph representing a factor with analytic evaluation
    of its Jacobian

    Parameters
    ----------
    factor_jacobian
        the function being wrapped, which can also return its jacobian by
        using the keyword argument `_variables` which indicates which,
        if any, variables the function must return the Jacobians for,
        so for example, for a function with variables 'x' and 'y', the
        call signatures would look like so,

        z, (dz_dx, dz_dy) = factor_jacoban(x=x, y=y, _variables=('x', 'y'))
        z, (dz_dy, dz_dx) = factor_jacoban(x=x, y=y, _variables=('y', 'x'))
        z, (dz_dy,) = factor_jacoban(x=x, y=y, _variables=('y',))
        z, () = factor_jacoban(x=x, y=y, _variables=())
        z = factor_jacoban(x=x, y=y, _variables=None)

    name: optional, str
        the name of the factor, if not passed then uses the name
        of the function passed

    vectorised: optional, bool
        if true the factor will call it the function directly over multiple
        inputs. If false the factor will call the function iteratively over
        each argument.

    is_scalar: optional, bool
        if true the factor returns a scalar value. Note if multiple arguments
        are passed then a vector will still be returned

    kwargs: Variables
        Variables for each keyword argument for the function
    """

    def __init__(
        self,
        factor_jacobian: Callable,
        *,
        name="",
        vectorised=False,
        plates: Tuple[Plate, ...] = (),
        **kwargs: Variable,
    ):
        self.vectorised = vectorised
        self.is_scalar = bool(plates)
        self._set_factor(factor_jacobian)

        AbstractFactor.__init__(
            self, **kwargs, plates=plates, name=name or factor_jacobian.__name__
        )

    def __hash__(self) -> int:
        # TODO: might this break factor repetition somewhere?
        return hash(self._factor)

    def _call_factor(
        self,
        values: Dict[str, np.ndarray],
        variables: Optional[Tuple[str, ...]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, ...]]]:
        """
        Call the underlying function

        Parameters
        ----------
        args
            Positional arguments for the function
        kwargs
            Keyword arguments for the function

        Returns
        -------
        Value returned by the factor
        """
        if self.vectorised:
            return self._factor(**values, _variables=variables)

        """Some factors may not be vectorised to broadcast over
        multiple inputs

        this method checks whether multiple input values have been
        passed, and if so automatically loops over the inputs.
        If any of the inputs have initial dimension one, it repeats
        that value to match the length of the other inputs

        If the other inputs do not match then it raises ValueError
        """
        kwargs_dims = {k: np.ndim(a) for k, a in values.items()}
        # Check dimensions of inputs directly match plates
        direct_call = all(dim == kwargs_dims[k] for k, dim in self._kwargs_dims.items())
        if direct_call:
            return self._factor(**values, _variables=variables)
        else:
            return self._multicall_factor(values, variables)

    def _multicall_factor(
        self,
        values: Dict[str, np.ndarray],
        variables: Optional[Tuple[str, ...]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, ...]]]:
        """call the factor multiple times and aggregates
        the results together
        """
        # Check dimensions of inputs match plates + 1
        vectorised = all(
            dim + 1 == np.ndim(values[k]) for k, dim in self._kwargs_dims.items()
        )

        if not vectorised:
            kwargs_dims = {k: np.ndim(a) for k, a in values.items()}
            raise ValueError(
                "input dimensions do not match required dims"
                f"input: **kwargs={kwargs_dims}"
                f"required: "
                f"**kwargs={self._kwargs_dims}"
            )

        kw_lens = {k: len(a) for k, a in values.items()}

        # checking 1st dimensions match
        sizes = set(kw_lens.values())
        dim0 = max(sizes)
        if sizes.difference({1, dim0}):
            raise ValueError(f"size mismatch first dimensions passed: {sizes}")

        iter_kws = {
            k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
            for k, a in values.items()
        }

        # TODO this loop can also be parallelised for increased performance
        fjacs = [
            self._factor(
                **{k: next(a) for k, a in iter_kws.items()}, _variables=variables
            )
            for _ in range(dim0)
        ]

        if variables is None:
            res = np.array([fjac for fjac in fjacs])
            return res
        else:
            res = np.array([fjac[0] for fjac in fjacs])
            njac = len(fjacs[0][1])
            jacs = tuple(np.array([fjac[1][i] for fjac in fjacs]) for i in range(njac))

            return res, jacs

    def __call__(
        self,
        variable_dict: Dict[Variable, np.ndarray],
        axis: Axis = None,
    ) -> FactorValue:
        values = self.resolve_variable_dict(variable_dict)
        val = self._call_factor(values, variables=None)
        val = aggregate(val, axis)
        return FactorValue(val, {})

    def func_jacobian(
        self,
        variable_dict: Dict[Variable, np.ndarray],
        variables: Optional[Tuple[Variable, ...]] = None,
        axis: Axis = None,
        **kwargs,
    ) -> Tuple[FactorValue, JacobianValue]:
        """
        Call the underlying factor

        Parameters
        ----------
        variable_dict :
            the values to call the function with
        variables : tuple of Variables
            the variables to calculate gradients and Jacobians for
            if variables = None then differentiates wrt all variables
        axis : None or False or int or tuple of ints, optional
            the axes to reduce the result over.
            if axis = None the sum over all dimensions
            if axis = False then does not reduce result

        Returns
        -------
        FactorValue, JacobianValue
            encapsulating the result of the function call
        """
        if variables is None:
            variables = self.variables

        variable_names = tuple(self._variable_name_kw[v.name] for v in variables)
        kwargs = self.resolve_variable_dict(variable_dict)
        val, jacs = self._call_factor(kwargs, variables=variable_names)
        grad_axis = tuple(range(np.ndim(val))) if axis is None else axis

        fval = FactorValue(aggregate(self._reshape_factor(val, kwargs), axis))
        fjac = JacobianValue(
            {
                v: FactorValue(aggregate(jac, grad_axis))
                for v, jac in zip(variables, jacs)
            }
        )
        return fval, fjac

    def __eq__(self, other: Union["Factor", Variable]):
        """
        If set equal to a variable that variable is taken to be deterministic and
        so a DeterministicFactorNode is generated.
        """
        if isinstance(other, Factor):
            if isinstance(other, type(self)):
                return (
                    (self._factor == other._factor)
                    and (
                        frozenset(self._kwargs.items())
                        == frozenset(other._kwargs.items())
                    )
                    and (frozenset(self.variables) == frozenset(other.variables))
                    and (
                        frozenset(self.deterministic_variables)
                        == frozenset(self.deterministic_variables)
                    )
                )
            else:
                return False

        return DeterministicFactorJacobian(self._factor, other, **self._kwargs)

    def __repr__(self) -> str:
        args = ", ".join(chain(map("{0[0]}={0[1]}".format, self._kwargs.items())))
        return f"FactorJacobian({self.name}, {args})"


class DeterministicFactorJacobian(FactorJacobian):
    """
    A deterministic factor is used to convert a function f(g(x)) to f(y)g(x) (integrating over y wit
    a delta function) so that it can be represented in a factor graph.

    Parameters
    ----------
    factor
        The original factor to which the deterministic factor is associated
    variable
        The deterministic variable that is returned by the factor, so
        to represent the case f(g(x)), we would define,

        ```
        >>> x = Variable('x')
        >>> y = Variable('y')
        >>> g_ = Factor(g, x) == y
        >>> f_ = Factor(f, y)
        ```
        Alternatively g could be directly defined,
        ```
        >>> g_ = DeterministicFactor(g, y, x=x)
        ```

    kwargs
        Variables for the original factor
    """

    def __init__(
        self,
        factor_jacobian: Callable,
        variable: Variable,
        vectorised=False,
        **kwargs: Variable,
    ):

        super().__init__(factor_jacobian, vectorised=vectorised, plates=(), **kwargs)
        self._deterministic_variables = (variable,)

    @property
    def deterministic_variables(self):
        return set(self._deterministic_variables)

    def func_jacobian(
        self,
        variable_dict: Dict[Variable, np.ndarray],
        variables: Optional[Tuple[Variable, ...]] = None,
        axis: Axis = None,
        **kwargs,
    ) -> Tuple[FactorValue, JacobianValue]:
        """
        Call this factor with a set of arguments

        Parameters
        ----------
        args
            Positional arguments for the underlying factor
        kwargs
            Keyword arguments for the underlying factor

        Returns
        -------
        An object encapsulating the value for the factor
        """
        if variables is None:
            variables = self.variables

        variable_names = tuple(self._variable_name_kw[v.name] for v in variables)
        kwargs = self.resolve_variable_dict(variable_dict)
        vals, *jacs = self._call_factor(kwargs, variables=variable_names)

        var_shapes = {self._kwargs[v]: np.shape(x) for v, x in kwargs.items()}
        shift, plate_sizes = self._plate_sizes(
            **{k: np.shape(x) for k, x in kwargs.items()}
        )
        start_plate = (None,) if shift else ()
        det_shapes = {
            v: tuple(plate_sizes[p] for p in start_plate + v.plates)
            for v in self.deterministic_variables
        }

        if not (isinstance(vals, tuple) or self.n_deterministic > 1):
            vals = (vals,)

        log_val = 0.0
        # log_val = (
        #     0. if (shape == () or axis is None) else
        #     aggregate(np.zeros(tuple(1 for _ in shape)), axis))
        det_vals = {
            k: np.reshape(val, det_shapes[k]) if det_shapes[k] else val
            for k, val in zip(self._deterministic_variables, vals)
        }
        fval = FactorValue(log_val, det_vals)

        vjacs = {}
        for k, _jacs in zip(self._deterministic_variables, jacs):
            for v, jac in zip(variables, _jacs):
                vjacs.setdefault(v, {})[k] = np.reshape(
                    jac, det_shapes[k] + var_shapes[v][shift:]
                )
        fjac = JacobianValue(
            {
                v: FactorValue(np.zeros(np.shape(log_val) + var_shapes[v]), vjacs[v])
                for v in variables
            }
        )
        return fval, fjac

    def __call__(
        self,
        variable_dict: Dict[Variable, np.ndarray],
        axis: Axis = None,
    ) -> FactorValue:
        return self.func_jacobian(variable_dict, (), axis=axis)[0]

    def __repr__(self) -> str:
        factor_str = super().__repr__()
        var_str = ", ".join(
            sorted(variable.name for variable in self._deterministic_variables)
        )
        return f"({factor_str} == ({var_str}))"
