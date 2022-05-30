import inspect
import logging
from abc import ABC
from copy import copy

import numpy as np

from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior_model.abstract import AbstractPriorModel

logger = logging.getLogger(
    __name__
)


def retrieve_name(var):
    first_name = None
    frame = inspect.currentframe()
    while frame is not None:
        for name, value in frame.f_locals.items():
            if var is value:
                first_name = name
        frame = frame.f_back

    return first_name


class CompoundPrior(
    AbstractPriorModel,
    ArithmeticMixin,
    ABC
):
    cls = float

    def __init__(self, left, right):
        """
        Comprises objects that are to undergo some arithmetic
        operation after realisation.

        Parameters
        ----------
        left
            A prior, promise or float
        right
            A prior, promise or float
        """
        super().__init__()

        self._left_name = retrieve_name(left) or "left"
        self._right_name = retrieve_name(right) or "right"

        if self._left_name == "left":
            self._left_name = "left_"

        if self._right_name == "right":
            self._right_name = "right_"

        self._left = None
        self._right = None

        self.left = left
        self.right = right

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @left.setter
    def left(self, left):
        self._left = left
        setattr(self, self._left_name, left)

    @right.setter
    def right(self, right):
        self._right = right
        setattr(self, self._right_name, right)

    def gaussian_prior_model_for_arguments(self, arguments):
        new = copy(self)
        try:
            new.left = new.left.gaussian_prior_model_for_arguments(
                arguments
            )
        except AttributeError:
            pass
        try:
            new.right = new.right.gaussian_prior_model_for_arguments(
                arguments
            )
        except AttributeError:
            pass
        return new

    def left_for_arguments(
            self,
            arguments: dict
    ):
        """
        Instantiate the left object.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to values

        Returns
        -------
        A value for the left object
        """
        try:
            return self._left.instance_for_arguments(arguments, )
        except AttributeError:
            return self._left

    def right_for_arguments(
            self,
            arguments: dict
    ):
        """
        Instantiate the right object.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to values

        Returns
        -------
        A value for the right object
        """
        try:
            return self._right.instance_for_arguments(arguments, )
        except AttributeError:
            return self._right


class SumPrior(CompoundPrior):
    """
    The sum of two objects, computed after realisation.
    """

    def _instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) + self.right_for_arguments(
            arguments
        )

    def __str__(self):
        return f"{self._left} + {self._right}"


class MultiplePrior(CompoundPrior):
    """
    The multiple of two objects, computed after realisation.
    """

    def __str__(self):
        return f"{self._left} * {self._right}"

    def _instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) * self.right_for_arguments(
            arguments
        )


class DivisionPrior(CompoundPrior):
    """
    One object divided by another, computed after realisation
    """

    def _instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) / self.right_for_arguments(
            arguments
        )


class FloorDivPrior(CompoundPrior):
    """
    One object divided by another and floored, computed after realisation.
    """

    def _instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) // self.right_for_arguments(
            arguments
        )


class ModPrior(CompoundPrior):
    """
    The modulus of a pair of objects, computed after realisation.
    """

    def _instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) % self.right_for_arguments(
            arguments
        )


class PowerPrior(CompoundPrior):
    """
    One object to the power of another, computed after realisation.
    """

    def _instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) ** self.right_for_arguments(
            arguments
        )


class ModifiedPrior(
    AbstractPriorModel,
    ABC,
    ArithmeticMixin
):
    def __init__(self, prior):
        super().__init__()
        self._prior_name = retrieve_name(prior)

        if self._prior_name == "prior":
            self._prior_name = "prior_"

        self.prior = prior

    @property
    def prior(self):
        return getattr(self, self._prior_name)

    @prior.setter
    def prior(self, prior):
        setattr(self, self._prior_name, prior)

    def gaussian_prior_model_for_arguments(self, arguments):
        new = copy(self)
        try:
            new.prior = new.prior.gaussian_prior_model_for_arguments(
                arguments
            )
        except AttributeError:
            pass
        return new


class NegativePrior(ModifiedPrior):
    """
    The negation of an object, computed after realisation.
    """

    def _instance_for_arguments(self, arguments):
        return -self.prior.instance_for_arguments(arguments, )


class AbsolutePrior(ModifiedPrior):
    """
    The absolute value of an object, computed after realisation.
    """

    def _instance_for_arguments(self, arguments):
        return abs(self.prior.instance_for_arguments(arguments, ))


class Log(ModifiedPrior):
    """
    The natural logarithm of an object, computed after realisation.
    """

    def _instance_for_arguments(
            self,
            arguments
    ):
        return np.log(self.prior.instance_for_arguments(arguments, ))


class Log10(ModifiedPrior):
    """
    The base10 logarithm of an object, computed after realisation.
    """

    def _instance_for_arguments(
            self,
            arguments
    ):
        return np.log10(self.prior.instance_for_arguments(arguments, ))
