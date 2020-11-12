from abc import ABC, abstractmethod
from functools import wraps
from typing import \
    List, Tuple, Dict, cast, Collection, Set, NamedTuple, Optional, Union
from itertools import count

import numpy as np

from autofit.mapper.variable import Variable, Plate
from autofit.graphical.utils import FactorValue, JacobianValue, HessianValue
from autofit.graphical.factor_graphs.numerical import (
    numerical_func_jacobian, numerical_func_jacobian_hessian)


class AbstractNode(ABC):
    _deterministic_variables: Set[Variable] = frozenset()
    _factor: callable = None
    _id = count()

    def __init__(
            self,
            **kwargs: Variable
    ):
        """
        A node in a factor graph

        Parameters
        ----------
        args
            Positional arguments passed to the factor
        kwargs
            Key word arguments passed to the value
        """
        self._kwargs = kwargs
        self._variable_name_kw = {
            v.name: kw for kw, v in kwargs.items()}
        self.id = next(self._id)

    def resolve_variable_dict(
            self, variable_dict:Dict[Variable, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return {
            self._variable_name_kw[v.name]: x
            for v, x in variable_dict.items()
            if v.name in self._variable_name_kw}

    @property
    @abstractmethod
    def variables(self):
        pass

    @property
    def variable_names(self) -> Dict[str, Variable]:
        return {
            variable.name: variable
            for variable in self.variables
        }

    @property
    @abstractmethod
    def deterministic_variables(self):
        return self._deterministic_variables

    def __getitem__(self, item):
        try:
            return self._kwargs[
                item
            ]
        except KeyError:
            for variable in self.variables | self._deterministic_variables:
                if variable.name == item:
                    return variable
            raise AttributeError(
                f"No attribute {item}"
            )

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
        return [
            arg.name
            for arg
            in self._kwargs.values()
        ]

    @property
    def all_variables(self) -> Set[Variable]:
        """
        A dictionary of variables associated with this node
        """
        return self.variables | self.deterministic_variables

    def _broadcast(
            self,
            plate_inds: np.ndarray,
            value: np.ndarray
    ) -> np.ndarray:
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
            value, 
            np.arange(plate_inds.size) + shift, 
            np.argsort(plate_inds) + shift)
        return np.reshape(movedvalue, newshape)

    def _broadcast2d(
            self,
            plate_inds: np.ndarray,
            value: np.ndarray
    ) -> np.ndarray:
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
            np.r_[plate_order, plate_order + ndim])
        return np.reshape(movedvalue, newshape)


    @property
    def plates(self) -> Tuple[Plate]:
        """
        A tuple of the set of all plates in this graph, ordered by id
        """
        return tuple(sorted(set(
            cast(Plate, plate)
            for variable
            in self.all_variables
            for plate in variable.plates
        )))

    @property
    def ndim(self) -> int:
        """
        The number of plates contained within this graph's variables

        That is, the total dimensions of those variables.
        """
        return len(self.plates)

    def _match_plates(
            self,
            plates: Collection[Plate]
    ) -> np.ndarray:
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
        return hash((
            self._factor, 
            frozenset(self.variable_names.items()),
            frozenset(self._deterministic_variables),))

    _numerical_func_jacobian = numerical_func_jacobian
    _numerical_func_jacobian_hessian = numerical_func_jacobian_hessian
    func_jacobian = numerical_func_jacobian
    func_jacobian_hessian = numerical_func_jacobian_hessian

    def jacobian(
            self, 
            values: Dict[Variable, np.array],
            variables: Optional[Tuple[Variable, ...]] = None,
            axis: Optional[Union[bool, int, Tuple[int, ...]]] = False, 
            _eps: float = 1e-6,
            _calc_deterministic: bool = True ) -> JacobianValue:
        return self.func_jacobian(
            values, variables, axis, 
            _eps=_eps, _calc_deterministic=_calc_deterministic)[1]
            
    def hessian(
            self, 
            values: Dict[Variable, np.array],
            variables: Optional[Tuple[Variable, ...]] = None,
            axis: Optional[Union[bool, int, Tuple[int, ...]]] = False, 
            _eps: float = 1e-6,
            _calc_deterministic: bool = True ) -> HessianValue:
        return self.func_jacobian_hessian(
            values, variables, axis, 
            _eps=_eps, _calc_deterministic=_calc_deterministic)[2]