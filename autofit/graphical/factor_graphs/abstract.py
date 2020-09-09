from abc import ABC, abstractmethod
from functools import wraps
from typing import List, Tuple, cast, Collection, Set
from itertools import count

import numpy as np

from autofit.mapper.variable import Variable, Plate


def accept_variable_dict(func):
    @wraps(func)
    def wrapper(self, variable_dict=None, **kwargs):
        if variable_dict is not None:
            kwargs.update({
                variable.name if isinstance(
                    variable,
                    Variable
                ) else variable: array
                for variable, array
                in variable_dict.items()
            })
        return func(
            self,
            **kwargs
        )

    return wrapper


class AbstractNode(ABC):
    _deterministic_variables: Set[Variable]
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
        self.id = next(self._id)

    @property
    @abstractmethod
    def variables(self):
        pass

    @property
    def variable_names(self):
        return {
            variable.name
            for variable
            in self.variable_names
        }

    @property
    @abstractmethod
    def deterministic_variables(self):
        return self._deterministic_variables

    def __getattr__(self, item):
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
        args = ", ".join(self.arg_names)
        kws = ", ".join(map("{0[0]}={0[1]}".format, self.kwarg_names))
        call_strs = []
        if args:
            call_strs.append(args)
        if kws:
            call_strs.extend(['*', kws])
        call_str = ", ".join(call_strs)
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
        return self.variables | self._deterministic_variables

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

        assert shift in {0, 1}
        newshape = np.ones(self.ndim + shift, dtype=int)
        newshape[:shift] = shape[:shift]
        newshape[shift + plate_inds] = shape[shift:]

        return np.reshape(value, newshape)

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
    def __call__(self, **kwargs):
        pass
