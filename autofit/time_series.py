import copy
from abc import ABC, abstractmethod
from typing import List, Dict

from scipy.stats import stats

from autofit.mapper.model import ModelInstance


class TimeSeriesPath:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def __getattr__(self, item: str) -> "TimeSeriesPath":
        return TimeSeriesPath(self.keys + [item])

    def get_value(self, instance: ModelInstance) -> float:
        for key in self.keys:
            instance = getattr(instance, key)
        return instance

    def __eq__(self, other: float) -> "Equality":
        return Equality(self, other)


class Equality:
    def __init__(self, path: TimeSeriesPath, value: float):
        self.path = path
        self.value = value


class AbstractTimeSeries(ABC):
    def __init__(self, instances: List[ModelInstance]):
        self.instances = instances

    def __getattr__(self, item: str):
        return TimeSeriesPath([item])

    def _value_map(self, path: TimeSeriesPath) -> Dict[float, ModelInstance]:
        return {path.get_value(instance): instance for instance in self.instances}

    def __getitem__(self, item: Equality):
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
                    path, self._interpolate(x, y, item.value),
                )

        return new_instance

    @staticmethod
    @abstractmethod
    def _interpolate(x, y, value):
        pass


class LinearTimeSeries(AbstractTimeSeries):
    @staticmethod
    def _interpolate(x, y, value):
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        return slope * value + intercept
