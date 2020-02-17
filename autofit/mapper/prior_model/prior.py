import copy
import inspect
import math
import sys
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from scipy.special import erfcinv

from autoconf import conf
from autofit import exc
from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.attribute_pair import cast_collection, PriorNameValue, InstanceNameValue
from autofit.mapper.prior_model.deferred import DeferredArgument


class WidthModifier:
    def __init__(self, value):
        self.value = value

    @classmethod
    def name_of_class(cls) -> str:
        """
        A string name for the class, with the prior suffix removed.
        """
        return cls.__name__.replace("WidthModifier", "")

    @classmethod
    def from_dict(cls, width_modifier_dict):
        return width_modifier_type_dict[
            width_modifier_dict["type"]
        ](
            value=width_modifier_dict["value"]
        )

    @property
    def dict(self):
        return {
            "type": self.name_of_class(),
            "value": self.value
        }

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.value == other.value


class RelativeWidthModifier(WidthModifier):
    pass


class AbsoluteWidthModifier(WidthModifier):
    pass


class TuplePrior:
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
        for prior_tuple in self.prior_tuples:
            setattr(tuple_prior, prior_tuple.name, arguments[prior_tuple.prior])
        return tuple_prior

    def __getitem__(self, item):
        return self.prior_tuples[item][1]


class Prior(ModelObject, ABC):
    def __init__(
            self,
            lower_limit=0.0,
            upper_limit=1.0,
            width_modifier=None
    ):
        """
        An object used to mappers a unit value to an attribute value for a specific
        class attribute.

        Parameters
        ----------
        lower_limit: Float
            The lowest value this prior can return
        upper_limit: Float
            The highest value this prior can return
        """
        if lower_limit >= upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )
        super().__init__()
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.width_modifier = width_modifier

    def assert_within_limits(self, value):
        if not (self.lower_limit <= value <= self.upper_limit):
            raise exc.PriorLimitException(
                "The physical value {} for a prior "
                "was not within its limits {}, {}".format(
                    value, self.lower_limit, self.upper_limit
                )
            )

    @staticmethod
    def for_class_and_attribute_name(cls, attribute_name):
        prior_dict = conf.instance.prior_config.for_class_and_prior_name(
            cls,
            attribute_name
        )
        return Prior.from_dict(
            prior_dict
        )

    @property
    def width(self):
        return self.upper_limit - self.lower_limit

    @abstractmethod
    def value_for(self, unit: float) -> float:
        """
        Return a physical value for a value between 0 and 1 with the transformation
        described by this prior.

        Parameters
        ----------
        unit
            A hypercube value between 0 and 1.

        Returns
        -------
        A physical value.
        """

    def __eq__(self, other):
        try:
            return self.id == other.id
        except AttributeError:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return "<{} id={} lower_limit={} upper_limit={}>".format(
            self.__class__.__name__, self.id, self.lower_limit, self.upper_limit
        )

    @classmethod
    def from_dict(cls, prior_dict: dict) -> Union["Prior", DeferredArgument]:
        """
        Create a prior from a JSON representation

        Parameters
        ----------
        prior_dict
            A dictionary representation of a prior including a type (e.g. Uniform)
            and all constructor arguments.

        Returns
        -------
        An instance of a child of this class.
        """
        if prior_dict["type"] == "Constant":
            return prior_dict["value"]
        if prior_dict["type"] == "Deferred":
            return DeferredArgument()

        prior_dict = copy.deepcopy(prior_dict)
        if "width_modifier" in prior_dict:
            prior_dict["width_modifier"] = WidthModifier.from_dict(
                prior_dict["width_modifier"]
            )
        # noinspection PyProtectedMember
        return prior_type_dict[
            prior_dict["type"]
        ](
            **{
                key: value
                for key, value
                in prior_dict.items()
                if key != "type"
            }
        )

    @property
    def dict(self) -> dict:
        """
        A dictionary representation of this prior
        """
        prior_dict = {
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "type": self.name_of_class()
        }
        if self.width_modifier is not None:
            prior_dict["width_modifier"] = self.width_modifier.dict
        return prior_dict

    @classmethod
    def name_of_class(cls) -> str:
        """
        A string name for the class, with the prior suffix removed.
        """
        return cls.__name__.replace("Prior", "")


class GaussianPrior(Prior):
    """A prior with a gaussian distribution"""

    def __init__(
            self,
            mean,
            sigma,
            lower_limit=-math.inf,
            upper_limit=math.inf,
            width_modifier=None
    ):
        super().__init__(
            lower_limit,
            upper_limit,
            width_modifier=width_modifier
        )
        self.mean = mean
        self.sigma = sigma
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1
        Returns
        -------
        value: Float
            A value for the attribute biased to the gaussian distribution
        """
        return self.mean + (self.sigma * math.sqrt(2) * erfcinv(2.0 * (1.0 - unit)))

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return (
                "GaussianPrior, mean = " + str(self.mean) + ", sigma = " + str(self.sigma)
        )

    def __repr__(self):
        return (
            "<GaussianPrior id={} mean={} sigma={} "
            "lower_limit={} upper_limit={}>".format(
                self.id, self.mean, self.sigma, self.lower_limit, self.upper_limit
            )
        )

    @property
    def dict(self) -> dict:
        """
        A dictionary representation of this prior
        """
        prior_dict = super().dict
        return {
            **prior_dict,
            "mean": self.mean,
            "sigma": self.sigma
        }


class UniformPrior(Prior):
    """A prior with a uniform distribution between a lower and upper limit"""

    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1
        Returns
        -------
        value: Float
            A value for the attribute between the upper and lower limits
        """
        return self.lower_limit + unit * (self.upper_limit - self.lower_limit)

    @property
    def mean(self):
        return self.lower_limit + (self.upper_limit - self.lower_limit) / 2

    @mean.setter
    def mean(self, new_value):
        difference = new_value - self.mean
        self.lower_limit += difference
        self.upper_limit += difference

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return (
                "UniformPrior, lower_limit = "
                + str(self.lower_limit)
                + ", upper_limit = "
                + str(self.upper_limit)
        )


class LogUniformPrior(UniformPrior):
    """A prior with a uniform distribution between a lower and upper limit"""

    def value_for(self, unit):
        """

        Parameters
        ----------
        unit: Float
            A unit hypercube value between 0 and 1
        Returns
        -------
        value: Float
            A value for the attribute between the upper and lower limits
        """
        return 10.0 ** (
                np.log10(self.lower_limit)
                + unit * (np.log10(self.upper_limit) - np.log10(self.lower_limit))
        )

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return (
                "LogUniformPrior, lower_limit = "
                + str(self.lower_limit)
                + ", upper_limit = "
                + str(self.upper_limit)
        )


def make_type_dict(cls):
    return {
        obj.name_of_class(): obj
        for _, obj in inspect.getmembers(
            sys.modules[__name__]
        )
        if (
                inspect.isclass(obj)
                and issubclass(obj, cls)
                and obj != Prior
        )
    }


prior_type_dict = make_type_dict(
    Prior
)

width_modifier_type_dict = make_type_dict(
    WidthModifier
)
