import copy
from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline
from typing import List, Dict, cast

from scipy.stats import stats

from autoconf.dictable import Dictable
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


class AbstractInterpolator(Dictable, ABC):
    def __init__(self, instances: List[ModelInstance]):
        """
        A TimeSeries allows interpolation on any variable.

        For example, each instance may have an attribute t. Other attributes
        can be determined for any given value of t by interpolating their
        values for each instance in the time series.

        Parameters
        ----------
        instances
            A list of instances (e.g. best fits) each with all the same attributes
            and a value on which a time series may be built
        """
        self.instances = instances

    def __getattr__(self, item: str) -> InterpolatorPath:
        """
        Used to indicate which attribute is the time attribute.

        This attribute may be at any path in the model.

        e.g.
        instance.t
        instance.something.t

        Parameters
        ----------
        item
            The name of an attribute of the instance

        Returns
        -------
        A class that keeps track of which attributes have been addressed
        """
        return InterpolatorPath([item])

    def _value_map(self, path: InterpolatorPath) -> Dict[float, ModelInstance]:
        """
        Maps know values to corresponding instances for a given path

        Parameters
        ----------
        path
            A path to an attribute, e.g. time

        Returns
        -------
        A dictionary mapping values of that attribute for each instance to the corresponding
        instance
        """
        return {path.get_value(instance): instance for instance in self.instances}

    def __getitem__(self, item: Equality) -> ModelInstance:
        """
        Create an artificial model instance which has values interpolated
        for a given interpolation value.

        Parameters
        ----------
        item
            Indicates a value for a given attribute to which the instance should
            be interpolated

        Returns
        -------
        An artificial instance with values interpolated

        Examples
        --------
        # Each instance in the time_series has an attribute 't'
        time_series = af.LinearTimeSeries([instance_1, instance_2, instance_3])

        # We can now create an instance which has a value of t = 3.5 by interpolating
        instance = time_series[time_series.t == 3.5)

        # We can also interpolate on any arbitrary attribute of the instance
        instance = time_series[time_series.some.arbitrary.attribute == -1.0]
        """
        value_map = self._value_map(item.path)
        try:
            return value_map[item.value]
        except KeyError:
            value_map = self._value_map(item.path)
            x = sorted(value_map)

            instance = self.instances[0]
            new_instance = copy.copy(instance)

            for path, _ in instance.path_instance_tuples_for_class(float):
                y = [value_map[value].object_for_path(path) for value in x]

                new_instance = new_instance.replacing_for_path(
                    path, self._interpolate(x, cast(List[float], y), item.value),
                )

        new_instance.replacing_for_path(tuple(item.path.keys), item.value)

        return new_instance

    @staticmethod
    @abstractmethod
    def _interpolate(x: List[float], y: List[float], value: float) -> float:
        """
        Interpolate a given attribute to find its effective value at some time

        Parameters
        ----------
        x
            A list of times (or another series)
        y
            A list which one value for each time
        value
            The time for which we want an interpolated value for the attribute

        Returns
        -------
        An interpolated value for the attribute
        """


class LinearInterpolator(AbstractInterpolator):
    """
    Assume all attributes have a linear relationship with time
    """

    @staticmethod
    def _interpolate(x, y, value):
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        return slope * value + intercept


class SplineInterpolator(AbstractInterpolator):
    """
    Interpolate data with a piecewise cubic polynomial which is twice continuously differentiable
    """

    @staticmethod
    def _interpolate(x, y, value):
        f = CubicSpline(x, y)
        return f(value)
