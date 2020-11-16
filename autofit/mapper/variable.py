from itertools import chain, count
from typing import Optional

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
    __slots__ = ("name", "plates")

    def __init__(self, name: str = None, *plates: Plate):
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
        super().__init__()
        self.name = name or f"{self.__class__.__name__.lower()}_{self.id}"
        self.plates = plates

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
