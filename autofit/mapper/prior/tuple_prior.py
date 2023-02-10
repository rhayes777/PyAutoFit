import copy
from typing import List, Tuple, Union, Dict

from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.attribute_pair import (
    cast_collection,
    PriorNameValue,
    InstanceNameValue,
)
from .abstract import Prior

NameValue = Tuple[
    str,
    Union[Prior, float]
]


class TuplePrior(ModelObject):
    """
    A prior comprising one or more priors in a tuple
    """

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @cast_collection(PriorNameValue)
    def prior_tuples(self):
        """
        Returns
        -------
        priors: [(String, Prior)]
            A list of priors contained in this tuple
        """
        return list(filter(lambda t: isinstance(t[1], Prior), self.__dict__.items()))

    @property
    def priors(self):
        return [prior for _, prior in self.prior_tuples]

    @property
    def unique_prior_tuples(self):
        return self.prior_tuples

    @property
    @cast_collection(InstanceNameValue)
    def instance_tuples(self):
        """
        Returns
        -------
        instances: [(String, instance)]
            A list of instances
        """
        return list(
            sorted(
                filter(lambda t: isinstance(t[1], float), self.__dict__.items()),
                key=lambda tup: tup[0],
            )
        )

    def value_for_arguments(self, arguments):
        """
        Parameters
        ----------
        arguments: {Prior: float}
            A dictionary of arguments

        Returns
        -------
        tuple: (float,...)
            A tuple of float values
        """

        def convert(tup):
            if hasattr(tup, "prior"):
                return arguments[tup.prior]
            return tup.instance

        return tuple(
            map(
                convert,
                sorted(
                    self.prior_tuples + self.instance_tuples, key=lambda tup: tup.name
                ),
            )
        )

    def gaussian_tuple_prior_for_arguments(self, arguments):
        """
        Parameters
        ----------
        arguments: {Prior: float}
            A dictionary of arguments

        Returns
        -------
        tuple_prior: TuplePrior
            A new tuple prior with gaussian priors
        """
        tuple_prior = TuplePrior()
        for name, prior in self.prior_tuples:
            setattr(
                tuple_prior,
                name,
                arguments[prior]
            )
        for name, value in self.instance_tuples:
            setattr(
                tuple_prior,
                name,
                value
            )
        return tuple_prior

    @property
    def tuples(self) -> List[NameValue]:
        """
        The names and instances of all priors and constants ordered
        by their name.

        This means they are in the order they should be in the tuple.
        """
        return sorted(
            self.prior_tuples + self.instance_tuples,
            key=lambda t: t[0]
        )

    def _with_paths(
            self,
            tree: Dict[str, dict]
    ) -> "TuplePrior":
        """
        An instance of this tuple prior with only tuples with positions
        indicated in the tree dictionary.

        Note applying this twice will give unexpected results.
        """
        new = TuplePrior()
        for key in tree:
            key, value = self._get_key_value(key)
            setattr(new, key, value)
        return new

    def _without_paths(
            self,
            tree: Dict[str, dict]
    ) -> "TuplePrior":
        """
        An instance of this tuple prior without tuples with positions
        indicated in the tree dictionary.

        Note applying this twice will give unexpected results.
        """
        new = copy.deepcopy(self)
        for key in tree:
            key, value = self._get_key_value(key)
            delattr(new, key)
        return new

    def _get_key_value(
            self,
            key: Union[str, int]
    ) -> NameValue:
        """
        Retrieve a key and value by an attribute name or index which may
        be expressed as a string or integer.
        """
        try:
            return self.tuples[int(key)]
        except ValueError:
            return key, getattr(
                self, key
            )

    def __getitem__(self, item):
        return self._get_key_value(item)[1]
