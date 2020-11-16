from abc import ABC
from inspect import getfullargspec
from itertools import chain, repeat
from typing import Tuple, Dict, Union, Set, NamedTuple, Callable

import numpy as np

from autofit.graphical.factor_graphs.abstract import AbstractNode, accept_variable_dict
from autofit.graphical.factor_graphs.numerical import numerical_jacobian
from autofit.mapper.variable import Variable


class FactorValue(NamedTuple):
    """
    The value associated with some factor
    """

    log_value: np.ndarray
    deterministic_values: Dict[Variable, np.ndarray]


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
        """
        Dictionary mapping the names of deterministic variables to those variables
        """
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
        return self.id

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
    def __init__(
            self,
            factor: Callable,
            name=None,
            vectorised=False,
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

        self.__function_shape = None
        super().__init__(
            **kwargs,
            name=name or factor.__name__
        )

    jacobian = numerical_jacobian

    def __hash__(self) -> int:
        # TODO: might this break factor repetition somewhere?
        return hash(self._factor)

    def _function_shape(self, **kwargs) -> Tuple[int, ...]:
        """
        Calculates the expected function shape based on the variables
        """
        if self.__function_shape is None:
            kws = self._resolve_args(**kwargs)
            variables = dict(
                (self._kwargs[k], x) for k, x in kws.items())
            var_shapes = {v: np.shape(x) for v, x in variables.items()}
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

            self.__function_shape = tuple(shape)
        return self.__function_shape

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
        kws = self._resolve_args(
            **kwargs
        )

        if self.vectorised:
            return self._factor(**kws)
        return self._py_vec_call(**kws)

    def _py_vec_call(
            self,
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
            for i in range(dim0):
                yield {
                    k: next(a) for k, a in iter_kws.items()}

        # TODO this loop can also be parallelised for increased performance
        res = np.array([
            self._factor(**kws)
            for kws in gen_kwargs()])

        return res

    @accept_variable_dict
    def __call__(
            self,
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
        val = self._call_factor(**kwargs)
        return FactorValue(
            val.reshape(
                self._function_shape(
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

        from autofit.graphical.factor_graphs import DeterministicFactorNode
        return DeterministicFactorNode(
            self._factor,
            other,
            **self._kwargs
        )

    def __repr__(self) -> str:
        args = ", ".join(chain(
            map("{0[0]}={0[1]}".format, self._kwargs.items())))
        return f"Factor({self.name})({args})"
