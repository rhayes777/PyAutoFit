from abc import ABC, abstractmethod
from typing import Optional

from autofit.mapper.prior_model.assertion import (
    CompoundAssertion,
    GreaterThanLessThanEqualAssertion,
    GreaterThanLessThanAssertion,
    AbstractAssertion
)
from autofit.tools.pipeline import ResultsCollection


class AbstractAssertionPromise(ABC):
    def __init__(self, name: Optional[str] = None):
        """
        A promised assertion. This happens when two assertions are
        compared with inequalities. Values associated with the priors
        that populate the underlying promises are compared when models
        are instantiated.

        Parameters
        ----------
        name
            The name of this assertion, printed if it fails
        """
        self.name = name

    @abstractmethod
    def populate(self, results_collection: ResultsCollection) -> AbstractAssertion:
        """
        Populate the underlying promises and create an Assertion

        Parameters
        ----------
        results_collection
            A collection of previous results

        Returns
        -------
        An assertion
        """


class CompoundAssertionPromise(AbstractAssertionPromise):
    def __init__(
            self,
            assertion_1: "AbstractAssertionPromise",
            assertion_2: "AbstractAssertionPromise"
    ):
        """
        A pair of assertion promises. This occurs when an assertion is
        made on range, for example (a < b) < c

        Parameters
        ----------
        assertion_1
            A promised assertion
        assertion_2
            A promised assertion
        """
        super().__init__()
        self.assertion_1 = assertion_1
        self.assertion_2 = assertion_2

    def populate(self, results_collection: ResultsCollection) -> CompoundAssertion:
        """
        Populate the underlying promises and create an Assertion

        Parameters
        ----------
        results_collection
            A collection of previous results

        Returns
        -------
        An assertion
        """
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
        """
        An assertion comparing a promise with another object.

        e.g. promise < 1
        e.g. promise_1 < promise_2

        Parameters
        ----------
        lower
            The object or promise asserted to have a lower value at instantiation
        greater
            The object or promises asserted to have a greater value at instantation
        """
        super().__init__()
        self._lower = lower
        self._greater = greater

    def __gt__(self, other):
        # noinspection PyTypeChecker
        return CompoundAssertionPromise(self, self._lower > other)

    def __lt__(self, other):
        # noinspection PyTypeChecker
        return CompoundAssertionPromise(self, self._greater < other)

    def __ge__(self, other):
        # noinspection PyTypeChecker
        return CompoundAssertionPromise(self, self._lower >= other)

    def __le__(self, other):
        # noinspection PyTypeChecker
        return CompoundAssertionPromise(self, self._greater <= other)

    def lower(self, results_collection: ResultsCollection):
        """
        Create a prior from the lower value if it is a promise else return it

        Parameters
        ----------
        results_collection
            A collection of previous results

        Returns
        -------
        The lower prior or object
        """
        try:
            return self._lower.populate(
                results_collection
            )
        except AttributeError:
            return self._lower

    def greater(self, results_collection):
        """
        Create a prior from the greater value if it is a promise else return it

        Parameters
        ----------
        results_collection
            A collection of previous results

        Returns
        -------
        The greater prior or object
        """
        try:
            return self._greater.populate(
                results_collection
            )
        except AttributeError:
            return self._greater


class GreaterThanLessThanAssertionPromise(ComparisonAssertionPromise):
    def populate(self, results_collection: ResultsCollection) -> GreaterThanLessThanAssertion:
        """
        Populate the underlying promises and create an Assertion

        Parameters
        ----------
        results_collection
            A collection of previous results

        Returns
        -------
        An assertion
        """
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
    def populate(self, results_collection: ResultsCollection) -> GreaterThanLessThanEqualAssertion:
        """
        Populate the underlying promises and create an Assertion

        Parameters
        ----------
        results_collection
            A collection of previous results

        Returns
        -------
        An assertion
        """
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
