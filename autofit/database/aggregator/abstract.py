from abc import ABC, abstractmethod
from typing import List

from .. import model as m


class AbstractAggregator(ABC):
    """
    Abstract collection of historical fits
    """

    @property
    @abstractmethod
    def fits(self) -> List[m.Fit]:
        """
        All fits in the collection
        """

    def __iter__(self):
        return iter(
            self.fits
        )

    def __getitem__(self, item):
        return self.fits[0]

    def values(self, name: str) -> list:
        """
        Retrieve the value associated with each fit with the given
        parameter name

        Parameters
        ----------
        name
            The name of some pickle, such as 'samples'

        Returns
        -------
        A list of objects, one for each fit
        """
        return [
            fit[name]
            for fit
            in self
        ]

    def __len__(self):
        return len(self.fits)

    def __eq__(self, other):
        if isinstance(other, list):
            return self.fits == other
        return super().__eq__(other)

    def __repr__(self):
        return str(self.fits)
