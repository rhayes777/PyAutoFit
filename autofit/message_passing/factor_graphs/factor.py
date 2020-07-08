from itertools import chain, repeat
from typing import Tuple, Dict, Any, Union, Set, NamedTuple, Callable, Optional

import numpy as np

from autofit.message_passing.factor_graphs.abstract import AbstractNode
from autofit.message_passing.factor_graphs.numerical import numerical_jacobian
from autofit.message_passing.factor_graphs.variable import Variable


class FactorValue(NamedTuple):
    """
    The value associated with some factor
    """

    log_value: np.ndarray
    deterministic_values: Dict[str, np.ndarray]


class Factor:
    def __init__(
            self,
            factor: Callable,
            name: Optional[str] = None,
            vectorised: bool = True
    ):
        """
        A factor in the model. This is a function that has been decomposed
        from the overall model.

        Parameters
        ----------
        factor
            Some callable
        name
            The name of this factor (defaults to the name of the callable)
        vectorised
            Can this factor be computed in a vectorised manner?
        """
        self.factor = factor
        self.name = name or factor.__name__
        self.vectorised = vectorised

    def call_factor(self, *args, **kwargs):
        """
        Call the underlying function and return its value for some set of
        arguments
        """
        return self.factor(*args, **kwargs)

    def __call__(self, *args: Variable):
        from autofit.message_passing.factor_graphs import FactorNode
        """
        Create a node in the graph from this factor by passing it the variables
        it uses.

        Parameters
        ----------
        args
            The variables with which this factor is associated

        Returns
        -------
        A node in the factor graph
        """
        return FactorNode(self, *args)

    def __hash__(self):
        return hash((self.name, self.factor))


class FactorNode(AbstractNode):
    def __init__(
            self,
            factor: Factor,
            *args: Variable,
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
        super().__init__(
            *args,
            **kwargs
        )
        self._factor = factor
        self._deterministic_variables = dict()

    jacobian = numerical_jacobian

    @property
    def _args_dims(self) -> Tuple[int]:
        """
        The number of plates for each positional argument variable
        """
        return tuple(map(
            len, self._args
        ))

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
            name: self._match_plates(
                variable.plates
            )
            for name, variable
            in self.all_variables.items()
        }

    @property
    def n_deterministic(self) -> int:
        """
        How many deterministic variables are there associated with this node?
        """
        return len(self._deterministic_variables)

    def __hash__(self) -> int:
        return hash(self._factor)

    def _resolve_args(
            self,
            *args: np.ndarray,
            **kwargs: np.ndarray
    ) -> Tuple[Any, dict]:
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
        n_args = len(args)
        args = args + tuple(kwargs[v] for v in self.arg_names[n_args:])
        kws = {n: kwargs[v] for n, v in self.kwarg_names}
        return args, kws

    def _function_shape(self, *args, **kwargs) -> Tuple[int, ...]:
        """
        Calculates the expected function shape based on the variables
        """
        args, kws = self._resolve_args(*args, **kwargs)
        variables = {
            v: x
            for v, x
            in zip(self.arg_names, args)
        }
        variables.update(
            (self._kwargs[k], x) for k, x in kws.items())
        var_shapes = {v: np.shape(x) for v, x in variables.items()}
        var_dims_diffs = {
            v: len(s) - self.all_variables[v].ndim
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

        return tuple(shape)

    def _call_factor(
            self,
            *args: Tuple[np.ndarray, ...],
            **kwargs: Dict[str, np.ndarray]
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
        args, kws = self._resolve_args(
            *args,
            **kwargs
        )

        if self._factor.vectorised:
            return self._factor.call_factor(*args, **kws)
        return self._py_vec_call(*args, **kws)

    def _py_vec_call(
            self,
            *args:
            Tuple[np.ndarray, ...],
            **kwargs: np.ndarray
    ) -> np.ndarray:
        """Some factors may not be vectorised to broadcast over
        multiple inputs

        this method checks whether multiple input values have been
        passed, and if so automatically loops over the inputs.
        If any of the inputs have initial dimension one, it repeats
        that value to match the length of the other inputs

        If the other inputs do not match then it raises ValueError
        """
        arg_dims = tuple(map(np.ndim, args))
        kwargs_dims = {k: np.ndim(a) for k, a in kwargs.items()}
        # Check dimensions of inputs directly match plates
        direct_call = (
                self._args_dims == arg_dims and
                all(dim == kwargs_dims[k] for k, dim in self._kwargs_dims.items()))
        if direct_call:
            return self._factor.call_factor(*args, **kwargs)

        # Check dimensions of inputs match plates + 1
        vectorised = (
                (tuple(d + 1 for d in self._args_dims) == arg_dims) and
                all(dim + 1 == kwargs_dims[k]
                    for k, dim in self._kwargs_dims.items()))

        if not vectorised:
            raise ValueError(
                "input dimensions do not match required dims"
                f"input: *args={arg_dims}, **kwargs={kwargs_dims}"
                f"required: *args={self._args_dims}, "
                f"**kwargs={self._kwargs_dims}")

        lens = [len(a) for a in args]
        kw_lens = {k: len(a) for k, a in kwargs.items()}

        # checking 1st dimensions match
        sizes = set(chain(lens, kw_lens.values()))
        dim0 = max(sizes)
        if sizes.difference({1, dim0}):
            raise ValueError(
                f"size mismatch first dimensions passed: {sizes}")

        # teeing up iterators to generate arguments to factor calls
        zip_args = zip(*(
            a if length == dim0 else repeat(a[0])
            for a, length in zip(args, lens)))
        iter_kws = {
            k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
            for k, a in kwargs.items()}

        # iterator to generate keyword arguments
        def gen_kwargs():
            for i in range(dim0):
                yield {
                    k: next(a) for k, a in iter_kws.items()}

        # TODO this loop can also be parallelised for increased performance
        res = np.array([
            self._factor.call_factor(*args, **kws)
            for args, kws in zip(zip_args, gen_kwargs())])

        return res

    def __call__(
            self,
            *args: np.ndarray,
            **kwargs: np.ndarray
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
        val = self._call_factor(*args, **kwargs)
        return FactorValue(
            val.reshape(
                self._function_shape(
                    *args,
                    **kwargs
                )
            ), {}
        )

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
        broadcasts the value of a variable to match the specific shape
        of the factor

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

    def __eq__(self, other: Union["FactorNode", Variable]):
        """
        If set equal to a variable that variable is taken to be deterministic and
        so a DeterministicFactorNode is generated.
        """
        if isinstance(other, FactorNode):
            if isinstance(other, type(self)):
                return (
                        (self._factor == other._factor)
                        and (self._args == other._args)
                        and (frozenset(self._kwargs.items())
                             == frozenset(other._kwargs.items()))
                        and (frozenset(self.variables.items())
                             == frozenset(other.variables.items()))
                        and (frozenset(self.deterministic_variables.items())
                             == frozenset(self.deterministic_variables.items())))
            else:
                return False

        from autofit.message_passing.factor_graphs import DeterministicFactorNode
        return DeterministicFactorNode(
            self._factor,
            other,
            *self._args,
            **self._kwargs
        )

    def __mul__(self, other):
        """
        When two factors are multiplied together this creates a graph
        """
        from autofit.message_passing.factor_graphs.graph import FactorGraph
        return FactorGraph([self]) * other

    def __repr__(self) -> str:
        args = ", ".join(chain(
            self.arg_names,
            map("{0[0]}={0[1]}".format, self.kwarg_names)))
        return f"Factor({self._factor.name})({args})"

    @property
    def variables(self) -> Dict[str, Variable]:
        """
        Dictionary mapping the names of variables to those variables
        """
        return self._variables

    @property
    def deterministic_variables(self) -> Dict[str, Variable]:
        """
        Dictionary mapping the names of deterministic variables to those variables
        """
        return self._deterministic_variables

    @property
    def name(self):
        """
        The name of this factor
        """
        return self._factor.name

    def variables_difference(
            self,
            *args: np.ndarray,
            **kwargs: np.ndarray
    ) -> Set[str]:
        """
        Compute which variables are missing when determining the sequence of calls

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        args = self.arg_names[:len(args)]
        return (self._variables.keys() - args).difference(kwargs)
