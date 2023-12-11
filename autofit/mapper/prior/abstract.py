import itertools
import os
import random
from abc import ABC, abstractmethod
from copy import copy
from typing import Union, Tuple, Optional, Dict

from autoconf import conf
from autofit import exc
from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior.deferred import DeferredArgument
from autofit.mapper.variable import Variable

epsilon = 1e-14


class Prior(Variable, ABC, ArithmeticMixin):
    __database_args__ = ("lower_limit", "upper_limit", "id_")

    _ids = itertools.count()

    def __init__(self, message, lower_limit=0.0, upper_limit=1.0, id_=None):
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
        super().__init__(id_=id_)
        self.message = message
        message.id_ = self.id

        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)
        if self.lower_limit >= self.upper_limit:
            raise exc.PriorException(
                "The upper limit of a prior must be greater than its lower limit"
            )

        self.width_modifier = None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Create a prior from a flattened PyTree

        Parameters
        ----------
        aux_data
            Auxiliary information that remains unchanged including
            the keys of the dict
        children
            Child objects subject to change

        Returns
        -------
        An instance of this class
        """
        return cls(*children, id_=aux_data[0])

    @property
    def lower_unit_limit(self) -> float:
        """
        The lower limit for this prior in unit vector space
        """
        return self.unit_value_for(self.lower_limit)

    @property
    def upper_unit_limit(self) -> float:
        """
        The upper limit for this prior in unit vector space
        """
        return self.unit_value_for(self.upper_limit)

    def unit_value_for(self, physical_value: float) -> float:
        """
        Compute the unit value between 0 and 1 for the physical value.
        """
        return self.message.cdf(physical_value)

    def with_message(self, message):
        new = copy(self)
        new.message = message
        return new

    def new(self):
        """
        Returns a copy of this prior with a new id assigned making it distinct
        """
        new = copy(self)
        new.id = next(self._ids)
        return new

    def with_limits(self, lower_limit: float, upper_limit: float) -> "Prior":
        """
        Create a new instance of the same prior class with the passed limits.
        """
        new = self.__class__(
            lower_limit=max(lower_limit, self.lower_limit),
            upper_limit=min(upper_limit, self.upper_limit),
        )
        new.message = self.message
        return new

    @property
    def factor(self):
        """
        A callable PDF used as a factor in factor graphs
        """
        return self.message.factor

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

    def random(
        self,
        lower_limit=0.0,
        upper_limit=1.0,
    ) -> float:
        """
        A random value sampled from this prior
        """
        return self.value_for(
            random.uniform(
                max(lower_limit, self.lower_unit_limit),
                min(upper_limit, self.upper_unit_limit),
            )
        )

    def value_for(self, unit: float, ignore_prior_limits=False) -> float:
        """
        Return a physical value for a value between 0 and 1 with the transformation
        described by this prior.

        Parameters
        ----------
        unit
            A unit value between 0 and 1.

        Returns
        -------
        A physical value, mapped from the unit value accoridng to the prior.
        """
        result = self.message.value_for(unit)
        if not ignore_prior_limits:
            self.assert_within_limits(result)
        return result

    def instance_for_arguments(self, arguments):
        return arguments[self]

    def project(self, samples, weights):
        result = copy(self)
        result.message = self.message.project(
            samples=samples,
            log_weight_list=weights,
            id_=self.id,
            lower_limit=self.lower_limit,
            upper_limit=self.upper_limit,
        )
        return result

    def __getattr__(self, item):
        if item in ("__setstate__", "__getstate__"):
            raise AttributeError(item)
        return getattr(self.message, item)

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

    def __str__(self):
        """The line of text describing this prior for the model_mapper.info file"""
        return f"{self.__class__.__name__} [{self.id}], {self.parameter_string}"

    @property
    @abstractmethod
    def parameter_string(self) -> str:
        pass

    def __float__(self):
        return self.value_for(0.5)

    @classmethod
    def from_dict(
        cls,
        prior_dict: dict,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[Dict[int, "Prior"]] = None,
    ) -> Union["Prior", DeferredArgument]:
        """
        Returns a prior from a JSON representation.

        Parameters
        ----------
        prior_dict : dict
            A dictionary representation of a prior including a type (e.g. Uniform) and all constructor arguments.
        reference
            A dictionary mapping prior ids to the priors they reference.
        loaded_ids
            A dictionary mapping prior ids to the priors they have loaded.

        Returns
        -------
        An instance of a child of this class.
        """
        loaded_ids = {} if loaded_ids is None else loaded_ids
        id_ = prior_dict.get("id")
        try:
            return loaded_ids[id_]
        except KeyError:
            pass

        if prior_dict["type"] == "Constant":
            return prior_dict["value"]
        if prior_dict["type"] == "Deferred":
            return DeferredArgument()

        from .uniform import UniformPrior
        from .log_uniform import LogUniformPrior
        from .gaussian import GaussianPrior
        from .log_gaussian import LogGaussianPrior

        prior_type_dict = {
            "Uniform": UniformPrior,
            "LogUniform": LogUniformPrior,
            "Gaussian": GaussianPrior,
            "LogGaussian": LogGaussianPrior,
        }

        # noinspection PyProtectedMember
        prior = prior_type_dict[prior_dict["type"]](
            **{
                key: value
                for key, value in prior_dict.items()
                if key not in ("type", "width_modifier", "gaussian_limits", "id")
            },
        )
        if id_ is not None:
            loaded_ids[id_] = prior
        return prior

    def dict(self) -> dict:
        """
        A dictionary representation of this prior
        """
        prior_dict = {
            "lower_limit": self.lower_limit,
            "upper_limit": self.upper_limit,
            "type": self.name_of_class(),
            "id": self.id,
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

    def gaussian_prior_model_for_arguments(self, arguments):
        return arguments[self]
