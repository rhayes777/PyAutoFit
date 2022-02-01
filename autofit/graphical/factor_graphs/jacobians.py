from itertools import repeat, chain
from typing import Tuple, Dict, Callable, Optional, Union, Any
from inspect import getfullargspec

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier

try:
    import jax

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

from autoconf import cached_property

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
    FactorValue,
)
from autofit.mapper.variable_operator import (
    RectVariableOperator,
    LinearOperator,
    VariableOperator,
)


def _is_variable(v, *args):
    return isinstance(v, Variable)


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
        return set(v[0] for v in nested_filter(_is_variable, self.factor_out))

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


class FactorJac(Factor):
    """
    Examples
    --------
    def linear(x, a, b):
        z = x.dot(a) + b
        return (z**2).sum(), z

    def likelihood(y, z):
        return ((y - z)**2).sum()

    def combined(x, y, a, b):
        like, z = linear(x, a, b)
        return like + likelihood(y, z)

    x_, a_, b_, y_, z_ = variables("x, a, b, y, z")
    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    values = {x_: x, y_: y, a_: a, b_: b}
    linear_factor = FactorVJP(
        linear, x_, a_, b_, factor_out=(FactorValue, z_))
    like_factor = FactorVJP(likelihood, y_, z_)
    full_factor = FactorVJP(combined, x_, y_, a_, b_)

    x = np.arange(10.).reshape(5, 2)
    a = np.arange(2.).reshape(2, 1)
    b = np.ones(1)
    y = np.arange(0., 10., 2).reshape(5, 1)
    values = {x_: x, y_: y, a_: a, b_: b}

    # Fully working problem
    fval, jac = full_factor.func_jacobian(values)
    grad = jac(1)

    linear_val, linear_jac = linear_factor.func_jacobian(values)
    like_val, like_jac = like_factor.func_jacobian(
        {**values, **linear_val.deterministic_values})
    combined_val = like_val + linear_val

    combined_grads = {FactorValue: 1.}
    for v, g in like_jac(combined_grads).items():
        combined_grads[v] = g + combined_grads.get(v, 0)

    for v, g in linear_jac(combined_grads).items():
        combined_grads[v] = g + combined_grads.get(v, 0)

    assert (fval.log_value - combined_val.log_value) == 0
    assert (grad - combined_grads).norm() == 0
    """

    def __init__(
        self,
        factor,
        *args: Variable,
        name="",
        factor_out=FactorValue,
        plates: Tuple[Plate, ...] = (),
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
        eps=1e-8,
    ):
        self.eps = eps
        self.args = args
        self.n_args = len(args)
        self.arg_names = [arg for arg in getfullargspec(factor).args]
        self.factor_out = factor_out

        kwargs = dict(zip(self.arg_names, self.args))
        name = name or factor.__name__
        AbstractFactor.__init__(self, **kwargs, name=name, plates=plates)

        det_variables = set(v[0] for v in nested_filter(_is_variable, factor_out))
        det_variables.discard(FactorValue)
        self._deterministic_variables = det_variables

        self._set_factor(factor)
        self._set_jacobians(
            vjp=vjp,
            factor_vjp=factor_vjp,
            factor_jacobian=factor_jacobian,
            jacobian=jacobian,
            numerical_jacobian=numerical_jacobian,
            jacfwd=jacfwd,
        )

    def _set_jacobians(
        self,
        vjp=False,
        factor_vjp=None,
        factor_jacobian=None,
        jacobian=None,
        numerical_jacobian=True,
        jacfwd=True,
    ):
        if vjp:
            if factor_vjp:
                self._factor_vjp = factor_vjp
            elif not _HAS_JAX:
                raise ModuleNotFoundError(
                    "jax needed if `factor_vjp` not passed with vjp=True"
                )
            else:
                self._factor_vjp = self._jax_factor_vjp

            self.func_jacobian = self._vjp_func_jacobian
        else:
            # This is set by default
            # self.func_jacobian = self._jvp_func_jacobian

            if factor_jacobian:
                self._factor_jacobian = factor_jacobian
            elif jacobian:
                self._jacobian = jacobian
            elif numerical_jacobian:
                self._factor_jacobian = self._numerical_factor_jacobian
            elif _HAS_JAX:
                if jacfwd:
                    self._jacobian = jax.jacfwd(self._factor, range(self.n_args))
                else:
                    self._jacobian = jax.jacobian(self._factor, range(self.n_args))

    def _factor_value(self, raw_fval):
        """Converts the raw output of the factor into a `FactorValue`
        where the values of the deterministic values are stored in a dict
        attribute `FactorValue.deterministic_values`
        """
        det_values = VariableData(
            nested_filter(_is_variable, self.factor_out, raw_fval)
        )
        fval = det_values.pop(FactorValue, 0.0)
        return FactorValue(fval, det_values)

    def __call__(self, values: VariableData):
        """Calls the factor with the values specified by the dictionary of
        values passed, returns a FactorValue with the value returned by the
        factor, and any deterministic factors"""
        raw_fval = self._factor(*(values[v] for v in self.args))
        return self._factor_value(raw_fval)

    def _jax_factor_vjp(self, *args) -> Tuple[Any, Callable]:
        return jax.vjp(self._factor, *args)

    def _vjp_func_jacobian(
        self, values: VariableData
    ) -> Tuple[FactorValue, VectorJacobianProduct]:
        """Calls the factor and returns the factor value with deterministic
        values, and a `VectorJacobianProduct` operator that allows the
        calculation of the gradient of the input values to be calculated
        with respect to the gradients of the output values (i.e backprop)
        """
        raw_fval, fvjp = self._factor_vjp(*(values[v] for v in self.args))
        fval = self._factor_value(raw_fval)

        fvjp_op = VectorJacobianProduct(
            self.factor_out,
            fvjp,
            *self.args,
            out_shapes=fval.to_dict().shapes,
        )
        return fval, fvjp_op

    def _jvp_func_jacobian(
        self, values: VariableData, **kwargs
    ) -> Tuple[FactorValue, JacobianVectorProduct]:
        args = (values[k] for k in self.args)
        raw_fval, raw_jac = self._factor_jacobian(*args, **kwargs)
        fval = self._factor_value(raw_fval)
        jvp = self._jac_out_to_jvp(raw_jac, values=fval.to_dict().merge(values))
        return fval, jvp

    func_jacobian = _jvp_func_jacobian

    def _factor_jacobian(self, *args, **kwargs) -> Tuple[Any, Any]:
        return self._factor(*args, **kwargs), self._jacobian(*args, **kwargs)

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
        # args = tuple(np.array(values[v]) for v in self.args)
        args = tuple(np.array(value) for value in args)

        raw_fval0 = self._factor(*args)
        fval0 = self._factor_value(raw_fval0).to_dict()
        # in_shapes = {v: np.shape(a) for v, a in values.items()}
        # out_shapes = {v: np.shape(a) for v, a in fval0.items()}

        jac = {
            # v0: {
            #     v1: np.empty_like(fval0[v0], shape=shape0 + shape1)
            #     for v1, shape1 in in_shapes.items()
            # }
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
                    fval1 = self._factor_value(self._factor(*args)).to_dict()
                    jac_v1_i = (fval1 - fval0) / eps
                    x_i -= eps
                    indexes = (Ellipsis,) + it.multi_index
                    for v0, jac_v0v_i in jac_v1_i.items():
                        jac[v0][i][indexes] = jac_v0v_i

        # This replicates the output of normal
        # jax.jacobian(self.factor, len(self.args))(*args)
        jac_out = nested_update(self.factor_out, jac)

        return raw_fval0, jac_out

    def _unpack_jacobian_out(self, raw_jac: Any) -> Dict[Variable, VariableData]:
        jac = {}
        for v0, vjac in nested_filter(_is_variable, self.factor_out, raw_jac):
            jac[v0] = VariableData()
            for v1, j in zip(self.args, vjac):
                jac[v0][v1] = j

        return jac

    def _jac_out_to_jvp(
        self, raw_jac: Any, values: VariableData
    ) -> JacobianVectorProduct:
        jac = self._unpack_jacobian_out(raw_jac)
        return JacobianVectorProduct.from_dense(jac, values=values)


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
    ) -> FactorValue:
        values = self.resolve_variable_dict(variable_dict)
        val = self._call_factor(values, variables=None)
        return FactorValue(val, {})

    def func_jacobian(
        self,
        variable_dict: Dict[Variable, np.ndarray],
        variables: Optional[Tuple[Variable, ...]] = None,
        **kwargs,
    ) -> Tuple[FactorValue, VariableData]:
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

        fval = FactorValue(val)
        fjac = VariableData({v: FactorValue(jac) for v, jac in zip(variables, jacs)})
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
        **kwargs,
    ) -> Tuple[FactorValue, VariableData]:
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
        fjac = VariableData(
            {
                v: FactorValue(np.zeros(np.shape(log_val) + var_shapes[v]), vjacs[v])
                for v in variables
            }
        )
        return fval, fjac

    def __call__(
        self,
        variable_dict: Dict[Variable, np.ndarray],
    ) -> FactorValue:
        return self.func_jacobian(variable_dict, ())[0]

    def __repr__(self) -> str:
        factor_str = super().__repr__()
        var_str = ", ".join(
            sorted(variable.name for variable in self._deterministic_variables)
        )
        return f"({factor_str} == ({var_str}))"
