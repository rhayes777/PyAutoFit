from abc import ABC

from autofit.mapper.prior.compound import CompoundPrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel


class ComparisonAssertion(CompoundPrior, ABC):
    def __init__(
            self,
            lower,
            greater,
            name=None
    ):
        super().__init__(lower, greater)
        self._name = name

    def __gt__(self, other):
        return CompoundAssertion(self, self.left > other)

    def __lt__(self, other):
        return CompoundAssertion(self, self.right < other)

    def __ge__(self, other):
        return CompoundAssertion(self, self.left >= other)

    def __le__(self, other):
        return CompoundAssertion(self, self.right <= other)


class GreaterThanLessThanAssertion(ComparisonAssertion):
    def _instance_for_arguments(self, arguments):
        """
        Assert that the value in the dictionary associated with the lower
        prior is lower than the value associated with the greater prior.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to physical values.

        Raises
        ------
        FitException
            If the assertion is not met
        """
        lower = self.left_for_arguments(arguments)
        greater = self.right_for_arguments(arguments)
        return lower < greater


class GreaterThanLessThanEqualAssertion(ComparisonAssertion):
    def _instance_for_arguments(self, arguments):
        """
        Assert that the value in the dictionary associated with the lower
        prior is lower than the value associated with the greater prior.

        Parameters
        ----------
        arguments
            A dictionary mapping priors to physical values.

        Raises
        ------
        FitException
            If the assertion is not met
        """
        return self.left_for_arguments(
            arguments
        ) <= self.right_for_arguments(
            arguments
        )


class CompoundAssertion(AbstractPriorModel):
    def __init__(
            self,
            assertion_1,
            assertion_2,
            name=None
    ):
        super().__init__()
        self.assertion_1 = assertion_1
        self.assertion_2 = assertion_2
        self._name = name

    def _instance_for_arguments(self, arguments):
        return self.assertion_1.instance_for_arguments(
            arguments
        ) and self.assertion_2.instance_for_arguments(
            arguments
        )


def unwrap(obj):
    try:
        return obj._value
    except AttributeError:
        return obj
