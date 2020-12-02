
from itertools import repeat, chain
from typing import \
    Tuple, Dict, List, Callable, Optional, Union
from functools import reduce 

import numpy as np

from autofit.mapper.variable import Variable
from autofit.graphical.factor_graphs.abstract import \
    FactorValue, JacobianValue
from autofit.graphical.factor_graphs.factor import \
    AbstractFactor, Factor, DeterministicFactor
from autofit.graphical.utils import \
    aggregate, Axis, cached_property


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
            name=None,
            vectorised=False,
            is_scalar=False, 
            **kwargs: Variable
    ):
        self.vectorised = vectorised
        self.is_scalar = is_scalar
        self._factor = factor_jacobian
        AbstractFactor.__init__(
            self, 
            **kwargs,
            name=name or factor_jacobian.__name__
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
        direct_call = (
            all(dim == kwargs_dims[k] for k, dim in self._kwargs_dims.items()))
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
            dim + 1 == np.ndim(values[k]) 
            for k, dim in self._kwargs_dims.items())

        if not vectorised:
            kwargs_dims = {k: np.ndim(a) for k, a in values.items()}
            raise ValueError(
                "input dimensions do not match required dims"
                f"input: **kwargs={kwargs_dims}"
                f"required: "
                f"**kwargs={self._kwargs_dims}")

        kw_lens = {k: len(a) for k, a in values.items()}

        # checking 1st dimensions match
        sizes = set(kw_lens.values())
        dim0 = max(sizes)
        if sizes.difference({1, dim0}):
            raise ValueError(
                f"size mismatch first dimensions passed: {sizes}")

        iter_kws = {
            k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
            for k, a in values.items()}

        # TODO this loop can also be parallelised for increased performance
        fjacs = [
            self._factor(**{
                    k: next(a) for k, a in iter_kws.items()}, 
                    _variables=variables)
             for _ in range(dim0)]
        
        if variables is None:
            res = np.array([fjac for fjac in fjacs])
            return res 
        else:
            res = np.array([fjac[0] for fjac in fjacs])
            njac = len(fjacs[0][1])
            jacs = tuple(
                np.array([fjac[1][i] for fjac in fjacs])
                for i in range(njac))

            return res, jacs

    def __call__(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> FactorValue:
        values = self.resolve_variable_dict(variable_dict)
        val = self._call_factor(values, variables=None)
        val = aggregate(val, axis)
        return FactorValue(val, {})

    def func_jacobian(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            variables: Optional[Tuple[Variable, ...]] = None,
            axis: Axis = False, 
            **kwargs
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

        variable_names = tuple(
            self._variable_name_kw[v.name]
            for v in variables)
        kwargs = self.resolve_variable_dict(variable_dict)
        val, jacs = self._call_factor(
            kwargs, variables=variable_names)
        grad_axis = tuple(range(np.ndim(val))) if axis is None else axis
        
        fval = FactorValue(
            aggregate(self._reshape_factor(val, kwargs), axis))
        fjac = {
            v: FactorValue(aggregate(jac, grad_axis))
            for v, jac in zip(variables, jacs)
        }
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
                        and (frozenset(self._kwargs.items())
                             == frozenset(other._kwargs.items()))
                        and (frozenset(self.variables)
                             == frozenset(other.variables))
                        and (frozenset(self.deterministic_variables)
                             == frozenset(self.deterministic_variables)))
            else:
                return False

        return DeterministicFactorJacobian(
            self._factor,
            other,
            **self._kwargs
        )

    def __repr__(self) -> str:
        args = ", ".join(chain(
            map("{0[0]}={0[1]}".format, self._kwargs.items())))
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
            is_scalar=False,
            **kwargs: Variable
    ):
        
        super().__init__(
            factor_jacobian,
            vectorised=vectorised,
            is_scalar=is_scalar, 
            **kwargs
        )
        self._deterministic_variables = variable, 

    @property
    def deterministic_variables(self):
        return set(self._deterministic_variables)

    def func_jacobian(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            variables: Optional[Tuple[Variable, ...]] = None,
            axis: Axis = False, 
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

        variable_names = tuple(
            self._variable_name_kw[v.name] for v in variables)
        kwargs = self.resolve_variable_dict(variable_dict)
        vals, *jacs = self._call_factor(
            kwargs, variables=variable_names)
        shift, shape = self._function_shape(**kwargs)
        plate_dim = dict(zip(self.plates, shape[shift:]))


        det_shapes = {
            v: shape[:shift] + tuple(
                plate_dim[p] for p in v.plates)
            for v in self.deterministic_variables
        }
        var_shapes = {
            self._kwargs[v]: np.shape(x) for v, x in kwargs.items()}

        if not (isinstance(vals, tuple) or self.n_deterministic > 1):
            vals = vals,

        log_val = (
            0. if (shape == () or axis is None) else 
            aggregate(np.zeros(tuple(1 for _ in shape)), axis))
        det_vals = {
            k: np.reshape(val, det_shapes[k])
            if det_shapes[k] else val
            for k, val in zip(self._deterministic_variables, vals)
        }
        fval = FactorValue(log_val, det_vals)

        vjacs = {}
        for k, _jacs in zip(self._deterministic_variables, jacs):
            for v, jac in zip(variables, _jacs):
                vjacs.setdefault(v, {})[k] = np.reshape(
                    jac, det_shapes[k] + var_shapes[v][shift:])
        fjac = {
            v: FactorValue(
                np.zeros(np.shape(log_val) + var_shapes[v]),
                vjacs[v])
            for v in variables
        }
        return fval, fjac

    def __call__(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> FactorValue:
        return self.func_jacobian(variable_dict, (), axis=axis)[0]

    def __repr__(self) -> str:
        factor_str = super().__repr__()
        var_str = ", ".join(sorted(variable.name for variable in self._deterministic_variables))
        return f"({factor_str} == ({var_str}))"