from abc import ABC
from collections import defaultdict, ChainMap, Counter
from itertools import chain, count, repeat
from typing import (
    NamedTuple, Callable, Tuple, Dict, Set, Union,
    Collection, Optional,
    Any
)

import numpy as np

from autofit.message_passing.factor_graphs.numerical import numerical_jacobian, numerical_hessdiag
from autofit.message_passing.utils import add_arrays


class Plate:
    _ids = count()

    def __init__(
            self,
            name: Optional[str] = None
    ):
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
        return isinstance(
            other,
            Plate
        ) and self.id == other.id

    def __hash__(self):
        return self.id


class Variable:
    __slots__ = ("name", "plates")

    def __init__(self, name: str, *plates):
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
        self.name = name
        self.plates = plates

    def __repr__(self):
        args = ", ".join(chain([self.name], map(repr, self.plates)))
        return f"{self.__class__.__name__}({args})"

    def __hash__(self):
        return hash((self.name, type(self)))

    @property
    def ndim(self) -> int:
        """
        How many dimensions does this variable have?
        """
        return len(self.plates)


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


class FactorValue(NamedTuple):
    """
    The value associated with some factor
    """

    log_value: np.ndarray
    deterministic_values: Dict[str, np.ndarray]


class AbstractNode(ABC):
    _deterministic_variables: Dict[str, Variable] = {}

    def __init__(self, *args, **kwargs):
        self._variables = {v.name: v for v in args}
        self._variables.update((v.name, v) for v in kwargs.values())

    @property
    def all_variables(self):
        return ChainMap(
            self._variables,
            self._deterministic_variables
        )

    def _broadcast(self, plate_inds: Collection[int], value: np.ndarray) -> np.ndarray:
        shape = np.shape(value)
        plate_inds = np.asanyarray(plate_inds)
        shift = len(shape) - plate_inds.size

        assert shift in {0, 1}
        newshape = np.ones(self.ndim + shift, dtype=int)
        newshape[:shift] = shape[:shift]
        newshape[shift + plate_inds] = shape[shift:]

        return np.reshape(value, newshape)

    def broadcast_plates(self, plates: Collection[Plate], value: np.ndarray) -> np.ndarray:
        return self._broadcast(self._match_plates(plates), value)

    @property
    def plates(self):
        return tuple(set(
            plate for v in self.all_variables.values() for plate in v.plates))

    @property
    def ndim(self):
        return len(self.plates)

    def _match_plates(self, plates: Collection[Plate]) -> np.ndarray:
        return np.array([self.plates.index(p) for p in plates], dtype=int)


class FactorNode(AbstractNode):
    def __init__(
            self,
            factor: Factor,
            *args: Variable,
            **kwargs: Variable
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self._factor = factor

        self._args = tuple(v.name for v in args)
        self._kwargs = {n: v.name for n, v in kwargs.items()}

    jacobian = numerical_jacobian
    hessdiag = numerical_hessdiag

    @property
    def _args_dims(self):
        return tuple(
            len(self.all_variables[v].plates) for v in self._args)

    @property
    def _kwargs_dims(self):
        return {
            k: len(self.all_variables[v].plates) for k, v in self._kwargs.items()
        }

    @property
    def _variable_plates(self):
        return {
            n: self._match_plates(v.plates)
            for n, v in self.all_variables.items()}

    @property
    def n_deterministic(self):
        return len(self._deterministic_variables)

    def __hash__(self) -> int:
        return hash(self._factor)

    def _resolve_args(self, *args: Tuple[np.ndarray, ...],
                      **kwargs: Dict[str, np.ndarray]
                      ) -> Tuple[Any, dict, Tuple[int, ...]]:
        """Transforms in the input arguments to match the arguments
        specified for the factor"""
        n_args = len(args)
        args = args + tuple(kwargs[v] for v in self._args[n_args:])
        kws = {n: kwargs[v] for n, v in self._kwargs}

        variables = {v: x for v, x in zip(self._args, args)}
        variables.update(
            (self._kwargs[k], x) for k, x in kws.items())
        return args, kws, self._function_shape(variables)

    def _function_shape(self, variables: Dict[str, np.ndarray]
                        ) -> Tuple[int, ...]:
        """Calculates the expected function shape based on the variables
        """
        var_shapes = {v: np.shape(x) for v, x in variables.items()}
        var_dims_diffs = {
            v: len(s) - self.all_variables[v].ndim  #
            for v, s in var_shapes.items()}
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

    def _variables_difference(self, *args: Tuple[np.ndarray, ...],
                              **kwargs: Dict[str, np.ndarray]
                              ) -> Set[str]:
        args = self._args[:len(args)]
        return (self._variables.keys() - args).difference(kwargs)

    def _call_factor(self, *args: Tuple[np.ndarray, ...],
                     **kwargs: Dict[str, np.ndarray]
                     ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        args, kws, shape = self._resolve_args(*args, **kwargs)
        if self._factor.vectorised:
            return self._factor.call_factor(*args, **kws), shape
        return self._py_vec_call(*args, **kws), shape

    def _py_vec_call(self, *args: Tuple[np.ndarray, ...],
                     **kwargs: Dict[str, np.ndarray]) -> np.ndarray:
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
            a if l == dim0 else repeat(a[0])
            for a, l in zip(args, lens)))
        iter_kws = {
            k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
            for k, a in kwargs.items()}

        # iterator to generate keyword arguments
        def gen_kwargs():
            for i in range(dim0):
                yield {
                    k: next(a) for k, a in iter_kws.items()}

        # TODO this loop can also be paralleised for increased performance
        res = np.array([
            self._factor.call_factor(*args, **kws)
            for args, kws in zip(zip_args, gen_kwargs())])

        return res

    def __call__(self, *args: Tuple[np.ndarray, ...],
                 **kwargs: Dict[str, np.ndarray]) -> FactorValue:
        val, shape = self._call_factor(*args, **kwargs)
        return FactorValue(val.reshape(shape), {})

    def broadcast_variable(self, variable: str, value: np.ndarray) -> np.ndarray:
        """
        broad casts the value of a variable to match the specific shape
        of the factor

        if the number of dimensions passed of the variable is 1
        greater than the dimensions of the variable then it's assumed
        that that dimension corresponds to multiple samples of that variable
        """
        return self._broadcast(self._variable_plates[variable], value)

    def collapse(self, variable: str, value: np.ndarray, agg_func=np.sum) -> np.ndarray:
        """
        broad casts the value of a variable to match the specific shape
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

    def __eq__(self, other: Union["FactorNode", Variable]
               ) -> Union[bool, "DeterministicFactorNode"]:
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

        elif isinstance(other, Variable):
            other = [other]

        return DeterministicFactorNode(
            self._factor, other,
            *(self._variables[name] for name in self._args),
            **{n: self._variables[name] for n, name in self._kwargs.items()})

    def __mul__(self, other) -> "FactorGraph":
        return FactorGraph([self]) * other

    def __repr__(self) -> str:
        args = ", ".join(chain(
            self._args,
            map("{0[0]}={0[1]}".format, self._kwargs.items())))
        return f"Factor({self._factor.name})({args})"

    @property
    def variables(self) -> Dict[str, Variable]:
        return self._variables

    @property
    def deterministic_variables(self) -> Dict[str, Variable]:
        return self._deterministic_variables

    @property
    def name(self):
        return self._factor.name

    @property
    def call_signature(self):
        args = ", ".join(self._args)
        kws = ", ".join(map("{0[0]}={0[1]}".format, self._kwargs.items()))
        call_strs = []
        if args:
            call_strs.append(args)
        if kws:
            call_strs.extend(['*', kws])
        call_str = ", ".join(call_strs)
        call_sig = f"{self.name}({call_str})"
        return call_sig


class DeterministicFactorNode(FactorNode):
    def __init__(self, factor: Factor,
                 deterministic_variables: Tuple[Variable, ...] = (),
                 *args: Tuple[Variable, ...],
                 **kwargs: Dict[str, Variable]):
        self._deterministic_variables = {v.name: v for v in deterministic_variables}
        super().__init__(factor, *args, **kwargs)

    def __call__(self, *args: Tuple[np.ndarray, ...],
                 **kwargs: Dict[str, np.ndarray]) -> FactorValue:
        res, shape = self._call_factor(*args, **kwargs)
        shift = len(shape) - self.ndim
        plate_dim = dict(zip(self.plates, shape[shift:]))

        det_shapes = {
            v: shape[:shift] + tuple(
                plate_dim[p] for v in self.deterministic_variables.values()
                for p in v.plates)
            for v in self.deterministic_variables}

        if not (isinstance(res, tuple) or self.n_deterministic > 1):
            res = res,

        log_val = 0. if shape == () else np.zeros(np.ones_like(shape))
        det_vals = {
            k: np.reshape(val, det_shapes[k]) if det_shapes[k] else val
            for k, val in zip(self._deterministic_variables, res)}
        return FactorValue(log_val, det_vals)

    def __repr__(self) -> str:
        factor_str = super().__repr__()
        var_str = ", ".join(self._deterministic_variables)
        return f"({factor_str} == ({var_str}))"


class FactorGraph(AbstractNode):
    def __init__(self, factors: Collection[FactorNode], name=None):
        super().__init__()
        self._factors = tuple(factors)
        self._name = ".".join(f.name for f in factors) if name is None else name

        self._variables = ChainMap(*(
            f.variables for f in self._factors))
        self._deterministic_variables = ChainMap(*(
            f.deterministic_variables for f in self._factors))
        self._all_variables = ChainMap(*(
            f.all_variables for f in self._factors))

        self._factor_variables = {
            f: f.variables for f in self._factors}
        self._factor_det_variables = {
            f: f.deterministic_variables for f in self._factors}
        self._factor_all_variables = {
            f: f.all_variables for f in self._factors}

        self._validate()
        self._hash = hash(frozenset(self.factors))

    @property
    def name(self):
        return self._name

    def _validate(self) -> None:
        det_var_counts = ", ".join(
            v for v, c in Counter(
                v for f in self.factors
                for v in f.deterministic_variables).items()
            if c > 1)
        if det_var_counts:
            raise ValueError(
                "Improper FactorGraph, "
                f"Deterministic variables {det_var_counts} appear in "
                "multiple factors")

        self._call_sequence, variables = self._get_call_sequence()
        self._all_factors = tuple(sum(self._call_sequence, []))

        diff = variables.keys() ^ self._variables.keys()
        if diff:
            raise ValueError(
                "Improper FactorGraph? unused variables: "
                + ", ".join(diff))

    def _get_call_sequence(self) -> None:
        """Calculates an appropriate call sequence for the factor graph

        each set of calls can be evaluated independently in parallel
        """
        variables = {
            v: self._variables[v] for v in
            (self._variables.keys() - self._deterministic_variables.keys())}

        factor_args = [factor._args for factor in self.factors]
        max_len = min(map(len, factor_args))
        self._args = tuple(
            factor_args[0][i] for i in range(max_len)
            if len(set(arg[i] for arg in factor_args)) == 1)
        self._kwargs = {k: k for k in variables.keys() - self._args}

        call_sets = defaultdict(list)
        for factor in self.factors:
            missing_vars = frozenset(factor._variables_difference(**variables))
            call_sets[missing_vars].append(factor)

        call_sequence = []
        while call_sets:
            # the factors that can be evaluated have no missing variables
            factors = call_sets.pop(frozenset(()))
            # if there's a KeyError then the FactorGraph is improper
            calls = []
            new_variables = {}
            for factor in factors:
                if isinstance(factor, DeterministicFactorNode):
                    det_vars = factor._deterministic_variables
                else:
                    det_vars = {}

                calls.append(factor)
                new_variables.update(det_vars)

            call_sequence.append(calls)

            # update to include newly calculated factors
            for missing in call_sets:
                if missing.intersection(new_variables):
                    factors = call_sets.pop(missing)
                    call_sets[missing.difference(new_variables)].extend(factors)

            variables.update(new_variables)
        return call_sequence, variables

    def __call__(self, *args: Tuple[np.ndarray, ...],
                 **kwargs: Dict[str, np.ndarray]) -> FactorValue:
        # generate set of factors to call, these are indexed by the
        # missing deterministic variables that need to be calculated
        log_value = 0.
        det_values = {}
        variables = kwargs

        n_args = len(args)
        if n_args > len(self._args):
            raise TypeError(
                f"too many arguments passed, must pass {len(self._args)} arguments, "
                f"factor graph call signature: {self.call_signature}")

        missing = self._kwargs.keys() - variables.keys() - set(self._args[:n_args])
        if missing:
            n_miss = len(missing)
            missing_str = ", ".join(missing)
            raise TypeError(f"{self} missing {n_miss} arguments: {missing_str}"
                            f"factor graph call signature: {self.call_signature}")

        for calls in self._call_sequence:
            # TODO parallelise this part?
            for factor in calls:
                ret = factor(*args, **variables)
                ret_value = self.broadcast_plates(factor.plates, ret.log_value)
                log_value = add_arrays(log_value, ret_value)
                det_values.update(ret.deterministic_values)
                variables.update(ret.deterministic_values)

        return FactorValue(log_value, det_values)

    def __mul__(self, other: FactorNode) -> "FactorGraph":
        factors = self.factors

        if isinstance(other, FactorGraph):
            factors += other.factors
        elif isinstance(other, FactorNode):
            factors += (other,)
        else:
            raise TypeError(
                f"type of passed element {(type(other))} "
                "does not match required types, (`FactorGraph`, `FactorNode`)")

        return type(self)(factors)

    def __repr__(self) -> str:
        factors_str = " * ".join(map(repr, self.factors))
        return f"({factors_str})"

    @property
    def factors(self) -> Tuple[FactorNode, ...]:
        return self._factors

    @property
    def factor_variables(self) -> Dict[FactorNode, str]:
        return self._factor_all_variables

    @property
    def factor_deterministic_variables(self) -> Dict[FactorNode, str]:
        return self._factor_det_variables

    @property
    def factor_all_variables(self) -> Dict[FactorNode, str]:
        return self._factor_all_variables
