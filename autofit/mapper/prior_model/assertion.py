from abc import ABC, abstractmethod

from autofit import exc


class AbstractAssertion(ABC):
    def __init__(self):
        self.name = None

    @abstractmethod
    def __call__(self, arg_dict: dict):
        """

        """


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
    def __call__(self, arg_dict: dict):
        """
        Assert that the value in the dictionary associated with the lower
        prior is lower than the value associated with the greater prior.

        Parameters
        ----------
        arg_dict
            A dictionary mapping priors to physical values.

        Raises
        ------
        FitException
            If the assertion is not met
        """
        if self.lower(arg_dict) > self.greater(arg_dict):
            raise exc.FitException(
                "Assertion failed" + ("" if self.name is None else f" '{self.name}'")
            )


class GreaterThanLessThanEqualAssertion(ComparisonAssertion):
    def __call__(self, arg_dict: dict):
        """
        Assert that the value in the dictionary associated with the lower
        prior is lower than the value associated with the greater prior.

        Parameters
        ----------
        arg_dict
            A dictionary mapping priors to physical values.

        Raises
        ------
        FitException
            If the assertion is not met
        """
        if not self.lower(arg_dict) <= self.greater(arg_dict):
            raise exc.FitException(
                "Assertion failed" + ("" if self.name is None else f" '{self.name}'")
            )


class CompoundAssertion(AbstractAssertion):
    def __init__(self, assertion_1, assertion_2):
        super().__init__()
        self.assertion_1 = assertion_1
        self.assertion_2 = assertion_2

    def __call__(self, arg_dict: dict):
        self.assertion_1(arg_dict)
        self.assertion_2(arg_dict)
