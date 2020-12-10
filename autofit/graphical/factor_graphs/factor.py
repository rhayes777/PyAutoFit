from abc import ABC
from inspect import getfullargspec
from itertools import chain, repeat
from typing import \
    Tuple, Dict, Union, Set, NamedTuple, Callable, Optional
from functools import lru_cache

import numpy as np

from autofit.graphical.utils import \
    aggregate, Axis
from autofit.graphical.factor_graphs.abstract import \
    AbstractNode, FactorValue, JacobianValue
from autofit.mapper.variable import Variable


class AbstractFactor(AbstractNode, ABC):
    def __init__(
            self,
            name=None,
            **kwargs: Variable,
    ):
        super().__init__(**kwargs)
        self._name = name or f"factor_{self.id}"
        self._deterministic_variables = set()

    @property
    def deterministic_variables(self) -> Set[Variable]:
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
        return {
            key: len(value)
            for key, value
            in self._kwargs.items()
        }

    @property
    def _variable_plates(self) -> Dict[str, np.ndarray]:
        """
        Maps the name of each variable to the indices of its plates
        within this node
        """
        return {
            variable: self._match_plates(
                variable.plates
            )
            for variable
            in self.all_variables
        }

    @property
    def n_deterministic(self) -> int:
        """
        How many deterministic variables are there associated with this node?
        """
        return len(self._deterministic_variables)

    def __hash__(self):
        return hash((type(self), self.id))

    def _resolve_args(
            self,
            **kwargs: np.ndarray
    ) -> dict:
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
    

class Factor(AbstractFactor):
    """
    A node in a graph representing a factor with analytic evaluation 
    of its Jacobian

    Parameters
    ----------
    factor
        the function being wrapped, must accept calls through keyword 
        argument

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

    Methods
    -------
    __call__({x: x0}, axis=axis)
        calls the factor, the passed input must be a dictionary with
        where the keys are the Variable objects that the function takes
        as input. The Variable keys only have to match the _names_
        of the variables of the function.  

        `axis` controls the shape of the output if the variables and factor
        have plates associated with them, when axis=False then 
        no reduction is performed, otherwise it is equivalent to calling
        np.sum(log_val, axis=axis) on the returned value
        
        returns a FactorValue object which behaves like an np.ndarray
        

    func_jacobian({x: x0}, variables=(x,), axis=axis)
        calls the factor and returns it value and the jacobian of its value
        with respect to the `variables` passed. if variables is None then
        it returns the jacobian with respect to all variables.

        returns fval, {x: d fval / dx}
    """
    def __init__(
            self,
            factor: Callable,
            name=None,
            vectorised=False,
            is_scalar=False,
            **kwargs: Variable
    ):
        """
        A node in a graph representing a factor

        Parameters
        ----------
        factor
            A wrapper around some callable
        args
            Variables representing positional arguments for the function
        kwargs
            Variables representing keyword arguments for the function
        """
        self.vectorised = vectorised
        self.is_scalar = is_scalar
        self._factor = factor

        args = getfullargspec(self._factor).args
        kwargs = {
            **kwargs,
            **{
                arg: Variable(arg)
                for arg
                in args
                if arg not in kwargs and arg != "self"
            }
        }


        super().__init__(
            **kwargs,
            name=name or factor.__name__
        )

    # jacobian = numerical_jacobian

    def __hash__(self) -> int:
        # TODO: might this break factor repetition somewhere?
        return hash(self._factor)

    def _reshape_factor(
            self, factor_val, values
    ):
        shift, shape = self._function_shape(**values)
        if self.is_scalar:
            if shift:
                return np.sum(
                    factor_val, axis=np.arange(1,np.ndim(factor_val)))
            else:
                return np.sum(factor_val)
        else:
            return np.reshape(factor_val, shape)

    def _function_shape(
            self, 
            **kwargs: np.ndarray) -> Tuple[int, Tuple[int, ...]]:
        """
        Calculates the expected function shape based on the variables
        """
        var_shapes = {
            k: np.shape(x) for k, x in kwargs.items()}
        return self._var_shape(**var_shapes)
    
    @lru_cache()
    def _var_shape(self, **kwargs: Tuple[int, ...]) -> Tuple[int, ...]:
        """This is called by _function_shape
        
        caches result so that does not have to be recalculated each call
        
        lru_cache caches f(x=1, y=2) to f(y=2, x=1), but in this case
        it should be find as the order of kwargs is set by self._kwargs
        which should be stable
        """
        var_shapes = {self._kwargs[k]: v for k, v in kwargs.items()}
        var_dims_diffs = {
            v: len(s) - v.ndim
            for v, s in var_shapes.items()
        }
        """
        If all the passed variables have an extra dimension then 
        we assume we're evaluating multiple instances of the function at the 
        same time

        otherwise an error is raised
        """
        if set(var_dims_diffs.values()) == {1}:
            # Check if we're passing multiple values e.g. for sampling
            shift = 1
        elif set(var_dims_diffs.values()) == {0}:
            shift = 0
        else:
            raise ValueError("dimensions of passed inputs do not match")

        """
        Updating shape of output array to match input arrays

        singleton dimensions are always assumed to match as in
        standard array broadcasting

        e.g. (1, 2, 3) == (3, 2, 1)
        """
        shape = np.ones(self.ndim + shift, dtype=int)
        for v, vs in var_shapes.items():
            ind = self._variable_plates[v] + shift
            vshape = vs[shift:]
            if shift:
                ind = np.r_[0, ind]
                vshape = (vs[0],) + vshape

            if shape.size:
                if not (
                        np.equal(shape[ind], 1) |
                        np.equal(shape[ind], vshape) |
                        np.equal(vshape, 1)).all():
                    raise AssertionError(
                        "Shapes do not match"
                    )
                shape[ind] = np.maximum(shape[ind], vshape)
        
        return shift, tuple(shape)

    def _call_factor(
            self,
            **kwargs: np.ndarray
    ) -> np.ndarray:
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
        # kws = self._resolve_args(
        #     **kwargs
        # )

        if self.vectorised:
            return self._factor(**kwargs)
            
        """Some factors may not be vectorised to broadcast over
        multiple inputs

        this method checks whether multiple input values have been
        passed, and if so automatically loops over the inputs.
        If any of the inputs have initial dimension one, it repeats
        that value to match the length of the other inputs

        If the other inputs do not match then it raises ValueError
        """
        kwargs_dims = {k: np.ndim(a) for k, a in kwargs.items()}
        # Check dimensions of inputs directly match plates
        direct_call = (
            all(dim == kwargs_dims[k] for k, dim in self._kwargs_dims.items()))
        if direct_call:
            return self._factor(**kwargs)

        # Check dimensions of inputs match plates + 1
        vectorised = (
            all(dim + 1 == kwargs_dims[k]
                for k, dim in self._kwargs_dims.items()))

        if not vectorised:
            raise ValueError(
                "input dimensions do not match required dims"
                f"input: **kwargs={kwargs_dims}"
                f"required: "
                f"**kwargs={self._kwargs_dims}")

        kw_lens = {k: len(a) for k, a in kwargs.items()}

        # checking 1st dimensions match
        sizes = set(kw_lens.values())
        dim0 = max(sizes)
        if sizes.difference({1, dim0}):
            raise ValueError(
                f"size mismatch first dimensions passed: {sizes}")

        iter_kws = {
            k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
            for k, a in kwargs.items()}

        # iterator to generate keyword arguments
        def gen_kwargs():
            for _ in range(dim0):
                yield {
                    k: next(a) for k, a in iter_kws.items()}

        # TODO this loop can also be parallelised for increased performance
        res = np.array([
            self._factor(**kws)
            for kws in gen_kwargs()])

        return res

    # @accept_variable_dict
    def __call__(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            axis: Axis = False, 
            # **kwargs: np.ndarray
    ) -> FactorValue:
        """
        Call the underlying factor

        Parameters
        ----------
        args
            Positional arguments for the factor
        kwargs
            Keyword arguments for the factor

        Returns
        -------
        Object encapsulating the result of the function call
        """
        kwargs = self.resolve_variable_dict(variable_dict)
        val = self._call_factor(**kwargs)
        val = aggregate(self._reshape_factor(val, kwargs), axis)
        return FactorValue(val, {})

    def broadcast_variable(
            self,
            variable: str,
            value: np.ndarray
    ) -> np.ndarray:
        """
        broadcasts the value of a variable to match the specific shape
        of the factor

        if the number of dimensions passed of the variable is 1
        greater than the dimensions of the variable then it's assumed
        that that dimension corresponds to multiple samples of that variable
        """
        return self._broadcast(
            self._variable_plates[variable],
            value
        )

    def collapse(
            self,
            variable: str,
            value: np.ndarray,
            agg_func=np.sum
    ) -> np.ndarray:
        """
        broadcasts `value` to match the specific shape of the factor,
        where `value` has the shape of the factor

        if the number of dimensions passed of the variable is 1
        greater than the dimensions of the variable then it's assumed
        that that dimension corresponds to multiple samples of that variable
        """
        ndim = np.ndim(value)
        shift = ndim - self.ndim
        assert shift in {0, 1}
        inds = self._variable_plates[variable] + shift
        dropaxes = tuple(np.setdiff1d(
            np.arange(shift, ndim), inds))

        # to ensured axes of returned array is in the correct order
        moved = np.moveaxis(value, inds, np.sort(inds))
        return agg_func(moved, axis=dropaxes)

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

        return DeterministicFactor(
            self._factor,
            other,
            **self._kwargs
        )

    def __repr__(self) -> str:
        args = ", ".join(chain(
            map("{0[0]}={0[1]}".format, self._kwargs.items())))
        return f"Factor({self.name}, {args})"


class DeterministicFactor(Factor):
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
            factor: Callable,
            variable: Variable,
            *args: Variable,
            **kwargs: Variable
    ):
        """
        A deterministic factor is used to convert a function f(g(x)) to f(y)g(x) (integrating over y wit
        a delta function) so that it can be represented in a factor graph.

        Parameters
        ----------
        factor
            The original factor to which the deterministic factor is associated
        variable
            The deterministic factor used
        args
            Variables for the original factor
        kwargs
            Variables for the original factor
        """
        super().__init__(
            factor,
            *args,
            **kwargs
        )
        self._deterministic_variables = {
            variable
        }

    def __call__(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            axis: Axis = False, 
            # **kwargs: np.ndarray
    ) -> FactorValue:
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
        kwargs = self.resolve_variable_dict(variable_dict)
        res = self._call_factor(**kwargs)
        shift, shape = self._function_shape(**kwargs)
        plate_dim = dict(zip(self.plates, shape[shift:]))

        det_shapes = {
            v: shape[:shift] + tuple(
                plate_dim[p] for p in v.plates)
            for v in self.deterministic_variables
        }

        if not (isinstance(res, tuple) or self.n_deterministic > 1):
            res = res,

        log_val = (
            0. if (shape == () or axis is None) else 
            aggregate(np.zeros(tuple(1 for _ in shape)), axis))
        det_vals = {
            k: np.reshape(val, det_shapes[k])
            if det_shapes[k]
            else val
            for k, val
            in zip(self._deterministic_variables, res)
        }
        return FactorValue(log_val, det_vals)

    def __repr__(self) -> str:
        factor_str = super().__repr__()
        var_str = ", ".join(sorted(variable.name for variable in self._deterministic_variables))
        return f"({factor_str} == ({var_str}))"
