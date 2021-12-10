from itertools import chain, count
from typing import Optional, Tuple 

import numpy as np

from autofit.mapper.model_object import ModelObject


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

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return self.id > other.id


class Variable(ModelObject):
    # __slots__ = ("name", "plates")

    def __init__(
            self,
            name: str = None,
            *plates: Plate,
            id_=None
    ):
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
        self.plates = plates
        super().__init__(
            id_=id_
        )
        self.name = name or f"{self.__class__.__name__.lower()}_{self.id}"

    def __repr__(self):
        args = ", ".join(chain([self.name], map(repr, self.plates)))
        return f"{self.__class__.__name__}({args})"

    def __hash__(self):
        return self.id

    def __len__(self):
        return len(self.plates)

    def __str__(self):
        return self.name

    @property
    def ndim(self) -> int:
        """
        How many dimensions does this variable have?
        """
        return len(self.plates)


def broadcast_plates(
        value: np.ndarray, 
        in_plates: Tuple[Plate, ...], 
        out_plates: Tuple[Plate, ...], 
        reducer: np.ufunc = np.sum
    ) -> np.ndarray:
    """
    Extract the indices of a collection of plates then match
    the shape of the data to that shape.

    Parameters
    ----------
    value
        A value to broadcast
    in_plates
        Plates representing the dimensions of the values
    out_plates
        Plates representing the output dimensions
    reducer
        function to reduce excess plates over, default np.sum
        must take axis as keyword argument


    Returns
    -------
    The value reshaped to match the plates
    """
    n_in = len(in_plates)
    n_out = len(out_plates)
    shift = np.ndim(value) - n_in
    if shift > 1 or shift < 0:
        raise ValueError("dimensions of value incompatible with passed plates")
    
    in_axes = list(range(shift, n_in + shift))
    out_axes = []
    k = n_out + shift
    
    for plate in in_plates:
        try:
            out_axes.append(out_plates.index(plate) + shift)
        except ValueError:
            out_axes.append(k)
            k += 1
            
    moved_value = np.moveaxis(
        np.expand_dims(value, tuple(range(n_in + shift, k))),
        in_axes,
        out_axes,
    )
    return reducer(moved_value, axis=tuple(range(n_out + shift, k)))