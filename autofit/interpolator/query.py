from typing import List
from autofit.mapper.model import ModelInstance


class InterpolatorPath:
    def __init__(self, keys: List[str]):
        """
        Addresses a given attribute in a ModelInstance

        Parameters
        ----------
        keys
            A list of attribute names
        """
        self.keys = keys

    def __hash__(self):
        return hash(tuple(self.keys))

    def __repr__(self):
        return f"InterpolatorPath({self.keys})"

    def __getattr__(self, item: str) -> "InterpolatorPath":
        """
        Add a new attribute name to the end of the path
        """
        return InterpolatorPath(self.keys + [item])

    def get_value(self, instance: ModelInstance) -> float:
        """
        Retrieve the value at the path for a given instance.

        Parameters
        ----------
        instance
            An instance of some model

        Returns
        -------
        The value for the instance at the given path.
        """
        for key in self.keys:
            instance = getattr(instance, key)
        return instance

    def __eq__(self, other: float) -> "Equality":
        """
        Create an object describing the value the addressed attribute
        should have for interpolation.

        Parameters
        ----------
        other
            A value to which the instance will be interpolated

        Returns
        -------
        An object describing how to interpolate
        """
        return Equality(self, other)


class Equality:
    def __init__(self, path: InterpolatorPath, value: float):
        """
        Describes the value of a given attribute for which other values
        are interpolated.

        Parameters
        ----------
        path
            The path to an attribute
        value
            The value that attribute should have
        """
        self.path = path
        self.value = value
