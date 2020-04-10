from abc import ABC

from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior_model.abstract import AbstractPriorModel


class CompoundPrior(
    AbstractPriorModel,
    ArithmeticMixin,
    ABC
):
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
        self.left = left
        self.right = right

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
            return self.left.instance_for_arguments(
                arguments
            )
        except AttributeError:
            return self.left

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
            return self.right.instance_for_arguments(
                arguments
            )
        except AttributeError:
            return self.right


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


class MultiplePrior(CompoundPrior):
    """
    The multiple of two objects, computed after realisation.
    """

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
        return -self.prior.instance_for_arguments(
            arguments
        )


class AbsolutePrior(ModifiedPrior):
    """
    The absolute value of an object, computed after realisation.
    """

    def _instance_for_arguments(self, arguments):
        return abs(self.prior.instance_for_arguments(
            arguments
        ))
