from itertools import count, chain
from typing import Optional, Callable


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
        from . import FactorNode
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
