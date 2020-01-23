import math
from functools import wraps

import numpy as np
from scipy.special import erfcinv

from autofit import conf
from autofit import exc
from autofit.mapper.model_object import ModelObject
from autofit.mapper.prior_model.deferred import DeferredArgument


def cast_collection(named_tuple):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return list(map(lambda tup: named_tuple(*tup), func(*args, **kwargs)))

        return wrapper

    return decorator


class AttributeNameValue(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __iter__(self):
        return iter(self.tuple)

    @property
    def tuple(self):
        return self.name, self.value

    def __getitem__(self, item):
        return self.tuple[item]

    def __eq__(self, other):
        if isinstance(other, AttributeNameValue):
            return self.tuple == other.tuple
        if isinstance(other, tuple):
            return self.tuple == other
        return False

    def __hash__(self):
        return hash(self.tuple)

    def __str__(self):
        return "({}, {})".format(self.name, self.value)

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, str(self))


class PriorNameValue(AttributeNameValue):
    @property
    def prior(self):
        return self.value


class instanceNameValue(AttributeNameValue):
    @property
    def instance(self):
        return self.value


class DeferredNameValue(AttributeNameValue):
    @property
    def deferred(self):
        return self.value


class TuplePrior(object):
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
    @cast_collection(instanceNameValue)
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


class Prior(ModelObject):
    """An object used to mappers a unit value to an attribute value for a specific
    class attribute """

    def __init__(self, lower_limit, upper_limit):
        if lower_limit >= upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )
        super().__init__()
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

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
        config_arr = conf.instance.prior_default.get_for_nearest_ancestor(
            cls, attribute_name
        )
        if config_arr[0] == "u":
            return UniformPrior(config_arr[1], config_arr[2])
        elif config_arr[0] == "n":
            return None
        elif config_arr[0] == "l":
            return LogUniformPrior(config_arr[1], config_arr[2])
        elif config_arr[0] == "g":
            limits = conf.instance.prior_limit.get_for_nearest_ancestor(
                cls, attribute_name
            )
            return GaussianPrior(config_arr[1], config_arr[2], *limits)
        elif config_arr[0] == "c":
            return config_arr[1]
        elif config_arr[0] == "d":
            return DeferredArgument()
        raise exc.PriorException(
            "Default prior for {} has no type indicator (u - Uniform, g - Gaussian, "
            "c - instance, d - Deferred)".format(attribute_name)
        )

    @property
    def width(self):
        return self.upper_limit - self.lower_limit

    def value_for(self, unit):
        raise NotImplementedError()

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


class GaussianPrior(Prior):
    """A prior with a gaussian distribution"""

    def __init__(self, mean, sigma, lower_limit=-math.inf, upper_limit=math.inf):
        super(GaussianPrior, self).__init__(lower_limit, upper_limit)
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


class UniformPrior(Prior):
    """A prior with a uniform distribution between a lower and upper limit"""

    def __init__(self, lower_limit=0.0, upper_limit=1.0):
        """

        Parameters
        ----------
        lower_limit: Float
            The lowest value this prior can return
        upper_limit: Float
            The highest value this prior can return
        """
        super(UniformPrior, self).__init__(lower_limit, upper_limit)

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

    def __init__(self, lower_limit=0.0, upper_limit=1.0):
        """

        Parameters
        ----------
        lower_limit: Float
            The lowest value this prior can return
        upper_limit: Float
            The highest value this prior can return
        """
        super(LogUniformPrior, self).__init__(lower_limit, upper_limit)

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
