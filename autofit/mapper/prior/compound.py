import inspect
import logging
from abc import ABC

from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior_model.abstract import AbstractPriorModel

logger = logging.getLogger(
    __name__
)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


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
        try:
            left_name = retrieve_name(left)
        except Exception as e:
            logger.info(e)
            left_name = "left"

        try:
            right_name = retrieve_name(right)
        except Exception as e:
            logger.info(e)
            right_name = "right"

        setattr(self, left_name, left)
        setattr(self, right_name, right)
        self._left = left
        self._right = right

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
        self.prior = prior


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
