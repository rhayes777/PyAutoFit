import copy
from typing import List

from scipy.stats import stats

from autofit.mapper.model import ModelInstance


class TimeSeriesPath:
    def __init__(self, keys):
        self.keys = keys

    def __getattr__(self, item):
        return TimeSeriesPath(self.keys + [item])

    def get_value(self, instance):
        for key in self.keys:
            instance = getattr(instance, key)
        return instance

    def __eq__(self, other):
        return Equality(self, other)


class Equality:
    def __init__(self, path: TimeSeriesPath, value: float):
        self.path = path
        self.value = value


class TimeSeries:
    def __init__(self, instances: List[ModelInstance]):
        self.instances = instances

    def __getattr__(self, item):
        return TimeSeriesPath([item])

    def _value_map(self, path):
        return {path.get_value(instance): instance for instance in self.instances}

    def __getitem__(self, item: Equality):
        return self._value_map(item.path)[item.value]


class LinearTimeSeries(TimeSeries):
    def __getitem__(self, item: Equality):
        try:
            return super().__getitem__(item)
        except KeyError:
            value_map = self._value_map(item.path)
            x = sorted(value_map)

            instance = self.instances[0]
            new_instance = copy.copy(instance)

            for path, _ in instance.path_instance_tuples_for_class(float):
                y = [value_map[value].object_for_path(path) for value in x]

                slope, intercept, r, p, std_err = stats.linregress(x, y)

                new_instance = new_instance.replacing_for_path(
                    path, slope * item.value + intercept,
                )

        return new_instance
