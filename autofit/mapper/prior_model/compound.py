from abc import ABC

from autofit.mapper.arithmetic import ArithmeticMixin
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

    def instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) + self.right_for_arguments(
            arguments
        )


class MultiplePrior(CompoundPrior):
    """
    The multiple of two objects, computed after realisation.
    """

    def instance_for_arguments(self, arguments):
        return self.left_for_arguments(
            arguments
        ) * self.right_for_arguments(
            arguments
        )


class NegativePrior(AbstractPriorModel, ArithmeticMixin):
    """
    The negation of an object, computed after realisation.
    """

    def __init__(self, prior):
        super().__init__()
        self.prior = prior

    def instance_for_arguments(self, arguments):
        return -self.prior.instance_for_arguments(
            arguments
        )
