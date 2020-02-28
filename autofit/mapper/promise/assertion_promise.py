from abc import ABC, abstractmethod

from autofit.mapper.prior_model.assertion import (
    CompoundAssertion,
    GreaterThanLessThanEqualAssertion,
    GreaterThanLessThanAssertion
)


class AbstractAssertionPromise(ABC):
    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def populate(self, results_collection):
        """

        Parameters
        ----------
        results_collection

        Returns
        -------

        """


class CompoundAssertionPromise(AbstractAssertionPromise):
    def __init__(self, assertion_1, assertion_2):
        super().__init__()
        self.assertion_1 = assertion_1
        self.assertion_2 = assertion_2

    def populate(self, results_collection):
        return CompoundAssertion(
            self.assertion_1.populate(
                results_collection
            ),
            self.assertion_2.populate(
                results_collection
            )
        )


class ComparisonAssertionPromise(AbstractAssertionPromise, ABC):
    def __init__(self, lower, greater):
        super().__init__()
        self._lower = lower
        self._greater = greater

    def __gt__(self, other):
        return CompoundAssertionPromise(self, self._lower > other)

    def __lt__(self, other):
        return CompoundAssertionPromise(self, self._greater < other)

    def __ge__(self, other):
        return CompoundAssertionPromise(self, self._lower >= other)

    def __le__(self, other):
        return CompoundAssertionPromise(self, self._greater <= other)

    def lower(self, results_collection):
        try:
            return self._lower.populate(
                results_collection
            )
        except AttributeError:
            return self._lower

    def greater(self, results_collection):
        try:
            return self._greater.populate(
                results_collection
            )
        except AttributeError:
            return self._greater


class GreaterThanLessThanAssertionPromise(ComparisonAssertionPromise):
    def populate(self, results_collection):
        lower = self.lower(
            results_collection
        )
        greater = self.greater(
            results_collection
        )
        return GreaterThanLessThanAssertion(
            lower=lower,
            greater=greater
        )


class GreaterThanLessThanEqualAssertionPromise(ComparisonAssertionPromise):
    def populate(self, results_collection):
        lower = self.lower(
            results_collection
        )
        greater = self.greater(
            results_collection
        )
        return GreaterThanLessThanEqualAssertion(
            lower=lower,
            greater=greater
        )
