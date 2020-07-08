from abc import ABC, abstractmethod
from collections import ChainMap
from typing import Union, Dict, List, Tuple, cast, Collection

import numpy as np

from autofit.message_passing.factor_graphs.variable import Variable, Plate


class AbstractNode(ABC):
    _deterministic_variables: Union[
        Dict[str, Variable],
        ChainMap
    ]

    def __init__(
            self,
            *args: Variable,
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
        self._variables = {
            v.name: v
            for v
            in args + tuple(
                kwargs.values()
            )
        }

        self._args = args
        self._kwargs = kwargs

    @property
    @abstractmethod
    def deterministic_variables(self):
        return self._deterministic_variables

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
    def arg_names(self) -> List[str]:
        """
        The names of the variables passed as positional arguments
        """
        return [
            arg.name
            for arg
            in self._args
        ]

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
    def all_variables(self) -> Dict[str, Variable]:
        """
        A dictionary of variables associated with this node
        """
        return {
            **self._variables,
            **self._deterministic_variables
        }

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
            in self.all_variables.values()
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
