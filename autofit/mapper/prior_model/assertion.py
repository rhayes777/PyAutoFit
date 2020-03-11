from abc import ABC

from autofit.mapper.prior_model.abstract import AbstractPriorModel


class AbstractAssertion(AbstractPriorModel, ABC):
    def __init__(self, name=None):
        super().__init__()
        self._name = name


class ComparisonAssertion(AbstractAssertion, ABC):
    def __init__(self, lower, greater):
        """
        Describes an assertion that the physical values associated with
        the lower and greater priors are lower and greater respectively.

        Parameters
        ----------
        lower: Prior
            A prior object with physical values that must be lower
        greater: Prior
            A prior object with physical values that must be greater
        """
        super().__init__()
        self._lower = lower
        self._greater = greater

    def __gt__(self, other):
        return CompoundAssertion(self, self._lower > other)

    def __lt__(self, other):
        return CompoundAssertion(self, self._greater < other)

    def __ge__(self, other):
        return CompoundAssertion(self, self._lower >= other)

    def __le__(self, other):
        return CompoundAssertion(self, self._greater <= other)

    def lower(self, arg_dict: dict):
        if isinstance(self._lower, float):
            return self._lower
        return arg_dict[self._lower]

    def greater(self, arg_dict: dict):
        if isinstance(self._greater, float):
            return self._greater
        return arg_dict[self._greater]


class GreaterThanLessThanAssertion(ComparisonAssertion):
    def instance_for_arguments(self, arguments):
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
        lower = self.lower(arguments)
        greater = self.greater(arguments)
        return lower < greater


class GreaterThanLessThanEqualAssertion(ComparisonAssertion):
    def instance_for_arguments(self, arguments):
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
        return self.lower(arguments) <= self.greater(arguments)


class CompoundAssertion(AbstractAssertion):
    def __init__(self, assertion_1, assertion_2):
        super().__init__()
        self.assertion_1 = assertion_1
        self.assertion_2 = assertion_2

    def instance_for_arguments(self, arguments):
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
