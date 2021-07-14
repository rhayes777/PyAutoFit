import inspect
import math
import random
import sys
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
from scipy import stats
from scipy.special import erfcinv

from autoconf import conf
from autofit import exc
from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior.deferred import DeferredArgument
from autofit.mapper.variable import Variable


class Limits:
    @staticmethod
    def for_class_and_attributes_name(cls, attribute_name):
        limit_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name, "gaussian_limits"]
        )
        return limit_dict["lower"], limit_dict["upper"]


class Prior(Variable, ABC, ArithmeticMixin):
    def __init__(self, lower_limit=0.0, upper_limit=1.0):
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
        super().__init__()
        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)
        if self.lower_limit >= self.upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )

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
        prior_dict = conf.instance.prior_config.for_class_and_suffix_path(
            cls, [attribute_name]
        )
        return Prior.from_dict(prior_dict)

    @property
    def width(self):
        return self.upper_limit - self.lower_limit

    def random(self) -> float:
        """
        A random value sampled from this prior
        """
        return self.value_for(
            random.random()
        )

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

    def instance_for_arguments(self, arguments):
        return arguments[self]

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

    def __float__(self):
        return self.value_for(0.5)

    @classmethod
    def from_dict(cls, prior_dict: dict) -> Union["Prior", DeferredArgument]:
        """
        Returns a prior from a JSON representation.

        Parameters
        ----------
        prior_dict : dict
            A dictionary representation of a prior including a type (e.g. Uniform) and all constructor arguments.

        Returns
        -------
        An instance of a child of this class.
        """
        if prior_dict["type"] == "Constant":
            return prior_dict["value"]
        if prior_dict["type"] == "Deferred":
            return DeferredArgument()

        # noinspection PyProtectedMember
        return prior_type_dict[prior_dict["type"]](
            **{
                key: value
                for key, value in prior_dict.items()
                if key not in ("type", "width_modifier", "gaussian_limits")
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
            "type": self.name_of_class(),
        }
        return prior_dict

    @classmethod
    def name_of_class(cls) -> str:
        """
        A string name for the class, with the prior suffix removed.
        """
        return cls.__name__.replace("Prior", "")

    @property
    def limits(self) -> Tuple[float, float]:
        return self.lower_limit, self.upper_limit


class GaussianPrior(Prior):
    """A prior with a gaussian distribution"""

    __name__ = "gaussian_prior"

    def __init__(self, mean, sigma, lower_limit=-math.inf, upper_limit=math.inf):
        super().__init__(lower_limit, upper_limit)
        self.mean = float(mean)
        self.sigma = float(sigma)

        self._log_pdf = None

    @property
    def logpdf(self):
        if self._log_pdf is None:
            norm = stats.norm(
                loc=self.mean,
                scale=self.sigma
            )
            self._log_pdf = norm.logpdf
        return self._log_pdf

    def __call__(self, x):
        return self.logpdf(x)

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

    def log_prior_from_value(self, value):
        """
        Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.

        This is used by Emcee in the log likelihood function evaluation.

        Parameters
        ----------
        value : float
            The physical value of this prior's corresponding parameter in a `NonLinearSearch` sample."""
        return (value - self.mean) ** 2.0 / (2 * self.sigma ** 2.0)

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
        return {**prior_dict, "mean": self.mean, "sigma": self.sigma}


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

    def log_prior_from_value(self, value):
        """
    Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.

        This is used by Emcee in the log likelihood function evaluation.

        NOTE: For a UniformPrior this is always zero, provided the value is between the lower and upper limit. Given
        this is check for when the instance is made (in the *instance_from_vector* function), we thus can simply return
        zero in this function.

        Parameters
        ----------
        value : float
            The physical value of this prior's corresponding parameter in a `NonLinearSearch` sample."""
        return 0.0

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

    def __init__(self, lower_limit=1e-6, upper_limit=1.0):
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
        super().__init__(lower_limit=lower_limit, upper_limit=upper_limit)
        if (self.lower_limit <= 0.0):
            raise exc.PriorException(
                "The lower limit of a LogUniformPrior cannot be zero or negative."
            )

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

    def log_prior_from_value(self, value):
        """
    Returns the log prior of a physical value, so the log likelihood of a model evaluation can be converted to a
        posterior as log_prior + log_likelihood.

        This is used by Emcee in the log likelihood function evaluation.

        Parameters
        ----------
        value : float
            The physical value of this prior's corresponding parameter in a `NonLinearSearch` sample."""
        return 1.0 / value

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
        for _, obj in inspect.getmembers(sys.modules[__name__])
        if (inspect.isclass(obj) and issubclass(obj, cls) and obj != Prior)
    }


prior_type_dict = make_type_dict(Prior)
