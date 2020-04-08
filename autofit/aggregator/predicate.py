from abc import ABC, abstractmethod
from typing import List, Iterator

from .phase_output import PhaseOutput


class AttributePredicate:
    def __init__(self, attribute: str):
        """
        Used to produce predicate objects for filtering in the aggregator.

        When an unrecognised attribute is called on an aggregator an instance
        of this object is created. This object implements comparison methods
        facilitating construction of predicates.

        Parameters
        ----------
        attribute
            The name of the attribute this predicate relates to.
        """
        self.attribute = attribute

    def __eq__(self, value):
        """
        Create a predicate which asks whether the given value is equal to
        the attribute of a phase.
        """
        return EqualityPredicate(
            self.attribute,
            value
        )

    def __ne__(self, other):
        """
        Create a predicate which asks whether the given value is not equal to
        the attribute of a phase.
        """
        return ~(self == other)

    def contains(self, value):
        """
        Create a predicate which asks whether the given is contained within
        the attribute of a phase.
        """
        return ContainsPredicate(
            self.attribute,
            value
        )


class AbstractPredicate(ABC):
    """
    Comparison between a value and some attribute of a phase
    """

    def filter(
            self,
            phases: List[PhaseOutput]
    ) -> Iterator[PhaseOutput]:
        """
        Only return phases for which this predicate evaluates to True

        Parameters
        ----------
        phases

        Returns
        -------

        """
        return filter(
            lambda phase: self(phase),
            phases
        )

    def __invert__(self) -> "NotPredicate":
        """
        A predicate that evaluates to True when this predicate evaluates
        to False
        """
        return NotPredicate(
            self
        )

    @abstractmethod
    def __call__(self, phase: PhaseOutput) -> bool:
        """
        Does the attribute of the phase match the requirement of this predicate?
        """


class ComparisonPredicate(AbstractPredicate, ABC):
    def __init__(
            self,
            attribute: str,
            value
    ):
        """
        Compare an attribute of a phase with a value.

        Parameters
        ----------
        attribute
            An attribute of a phase
        value
            A value to which the attribute is compared
        """
        self.attribute = attribute
        self.value = value


class ContainsPredicate(ComparisonPredicate):
    def __call__(
            self,
            phase: PhaseOutput
    ) -> bool:
        """
        Parameters
        ----------
        phase
            An object representing the output of a given phase.

        Returns
        -------
        True iff the value of the attribute of the phase contains
        the value associated with this predicate
        """
        return self.value in getattr(
            phase,
            self.attribute
        )


class EqualityPredicate(ComparisonPredicate):
    def __call__(self, phase):
        """
        Parameters
        ----------
        phase
            An object representing the output of a given phase.

        Returns
        -------
        True iff the value of the attribute of the phase is equal to
        the value associated with this predicate
        """
        return getattr(
            phase,
            self.attribute
        ) == self.value


class NotPredicate(AbstractPredicate):
    def __init__(
            self,
            predicate: AbstractPredicate
    ):
        """
        Negates the output of a predicate.

        If the predicate would have returned True for a given phase
        it now returns False and vice-versa.

        Parameters
        ----------
        predicate
            A predicate that is negated
        """
        self.predicate = predicate

    def __call__(self, phase: PhaseOutput) -> bool:
        """
        Evaluate the predicate for the phase and return the negation
        of the result.

        Parameters
        ----------
        phase
            The output of an AutoFit phase

        Returns
        -------
        The negation of the underlying predicate
        """
        return not self.predicate(
            phase
        )
