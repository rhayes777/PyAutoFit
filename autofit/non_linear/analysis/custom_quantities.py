from typing import List, Optional, Union

import numpy as np

from autofit import exc


class CustomQuantities:
    def __init__(
        self,
        names: Optional[List[str]] = None,
        values=None,
    ):
        """
        A collection of custom quantities computed during an optimisation.

        Parameters
        ----------
        names
            The names of the custom quantities.
        values
            The values of the custom quantities.
        """
        self.names = names or []
        self.values = values or []

        self._current_row = []

    def _position(self, name: str):
        """
        Get the position of a custom quantity by its name.

        Parameters
        ----------
        name
            The name of the custom quantity.

        Returns
        -------
        The column position of the custom quantity.
        """
        return self.names.index(name)

    def add(self, **kwargs: float):
        """
        Add custom quantities to the collection. This should be called once
        per a fit and must always be passed the same custom quantities.

        Parameters
        ----------
        kwargs
            The custom quantities to add.

        Raises
        ------
        SamplesException
            If the same custom quantities are not passed to `add` each iteration.
        """
        if self.names and set(kwargs.keys()) != set(self.names):
            raise exc.SamplesException(
                "The same custom quantities must be passed to `add` each iteration."
            )

        for name in kwargs:
            if name not in self.names:
                self.names.append(name)

        self.values.append([kwargs[name] for name in self.names])

    def efficient(self) -> "CustomQuantities":
        """
        Convert the values to a numpy array for efficient storage in the database.
        """
        return CustomQuantities(
            names=self.names,
            values=np.array(self.values),
        )

    def _convert_values_to_dict(self, values):
        return {name: value for name, value in zip(self.names, values)}

    def __getitem__(self, item: Union[int, str]) -> Union[dict, List[float]]:
        """
        Retrieve a dictionary of custom quantities given a row number or a list of
        values for a specific custom quantity given its name.

        Parameters
        ----------
        item
            The row number or name of the custom quantity.

        Returns
        -------
        The custom quantities.
        """
        if isinstance(item, str):
            return [value_list[self._position(item)] for value_list in self.values]
        return self._convert_values_to_dict(self.values[item])

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for values in self.values:
            yield self._convert_values_to_dict(values)
