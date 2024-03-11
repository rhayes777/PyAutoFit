import numpy as np

from autofit import exc


class CustomQuantities:
    def __init__(self, names=None, values=None):
        self.names = names or []
        self.values = values or []

        self._current_row = []

    def _position(self, name):
        return self.names.index(name)

    def add(self, **kwargs):
        try:
            for name in kwargs:
                if name not in self.names:
                    self.names.append(name)

            self.values.append([kwargs[name] for name in self.names])
        except KeyError:
            raise exc.SamplesException(
                "The same custom quantities must be added once each fit."
            )

    def efficient(self):
        return CustomQuantities(
            names=self.names,
            values=np.array(self.values),
        )

    def _convert_values_to_dict(self, values):
        return {name: value for name, value in zip(self.names, values)}

    def __getitem__(self, item):
        if isinstance(item, str):
            return [value_list[self._position(item)] for value_list in self.values]
        return self._convert_values_to_dict(self.values[item])

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for values in self.values:
            yield self._convert_values_to_dict(values)
