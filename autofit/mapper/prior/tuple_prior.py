import copy

from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.attribute_pair import (
    cast_collection,
    PriorNameValue,
    InstanceNameValue,
)
from .abstract import Prior


class TuplePrior(ModelObject):
    """
    A prior comprising one or more priors in a tuple
    """

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
    def tuples(self):
        return sorted(
            self.prior_tuples + self.instance_tuples,
            key=lambda t: t[0]
        )

    def _with_paths(self, tree):
        new = TuplePrior()
        for key in tree:
            name, value = self.tuples[int(key)]
            setattr(new, name, value)
        return new

    def _without_paths(self, tree):
        new = copy.deepcopy(self)
        for key in tree:
            name, value = self.tuples[int(key)]
            delattr(new, name)
        return new

    def __getitem__(self, item):
        tuples = sorted(
            self.prior_tuples + self.instance_tuples,
            key=lambda t: t[0]
        )
        return tuples[item][1]
