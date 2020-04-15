from abc import ABC, abstractmethod
from typing import List, Iterator

from .phase_output import PhaseOutput


class AttributePredicate:
    def __init__(self, *path):
        """
        Used to produce predicate objects for filtering in the aggregator.

        When an unrecognised attribute is called on an aggregator an instance
        of this object is created. This object implements comparison methods
        facilitating construction of predicates.

        Parameters
        ----------
        path
            A series of names of attributes that can be used to get a value.
            For example, (mask, pixel_size) would get the pixel size of a mask
            when evaluated for a given phase.
        """
        self.path = path

    def value_for_phase(
            self,
            phase: PhaseOutput
    ):
        """
        Recurse the phase output by iterating the attributes in the path
        and getting a value for each attribute.
        """
        value = phase
        for attribute in self.path:
            value = getattr(
                value,
                attribute
            )
        return value

    def __eq__(self, value):
        """
        Create a predicate which asks whether the given value is equal to
        the attribute of a phase.
        """
        return EqualityPredicate(
            self,
            value
        )

    def __le__(self, other):
        return OrPredicate(
            self == other,
            self < other
        )

    def __ge__(self, other):
        return OrPredicate(
            self == other,
            self > other
        )

    def __getattr__(self, item: str) -> "AttributePredicate":
        """
        Adds another item to the path
        """
        return AttributePredicate(
            *self.path, item
        )

    def __gt__(self, other):
        """
        Is the value of this attribute for a given phase greater than some
        other value?
        """
        return GreaterThanPredicate(
            self, other
        )

    def __lt__(self, other):
        """
        Is the value of this attribute for a given phase less than some
        other value?
        """
        return LessThanPredicate(
            self, other
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
            self,
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

    def __or__(self, other: "AbstractPredicate") -> "OrPredicate":
        """
        Returns a predicate that is true if either predicate is true
        for a given phase.
        """
        return OrPredicate(self, other)

    def __and__(self, other: "AbstractPredicate") -> "AndPredicate":
        """
        Returns a predicate that is true if both predicates are true
        for a given phase.
        """
        return AndPredicate(self, other)

    @abstractmethod
    def __call__(self, phase: PhaseOutput) -> bool:
        """
        Does the attribute of the phase match the requirement of this predicate?
        """


class CombinationPredicate(AbstractPredicate, ABC):
    def __init__(
            self,
            one: AbstractPredicate,
            two: AbstractPredicate
    ):
        """
        Abstract predicate combining two other predicates.

        Parameters
        ----------
        one
        two
            Child predicates
        """
        self.one = one
        self.two = two


class OrPredicate(CombinationPredicate):
    def __call__(self, phase: PhaseOutput):
        """
        The disjunction of two predicates.

        Parameters
        ----------
        phase
            An object representing the output of a given phase.

        Returns
        -------
        True if either predicate is True for the phase
        """
        return self.one(phase) or self.two(phase)


class AndPredicate(CombinationPredicate):
    def __call__(self, phase: PhaseOutput):
        """
        The conjunction of two predicates.

        Parameters
        ----------
        phase
            An object representing the output of a given phase.

        Returns
        -------
        True if both predicates are True for the phase
        """
        return self.one(phase) and self.two(phase)


class ComparisonPredicate(AbstractPredicate, ABC):
    def __init__(
            self,
            attribute_predicate: AttributePredicate,
            value
    ):
        """
        Compare an attribute of a phase with a value.

        Parameters
        ----------
        attribute_predicate
            An attribute path of a phase
        value
            A value to which the attribute is compared
        """
        self.attribute_predicate = attribute_predicate
        self._value = value

    def value(
            self,
            phase
    ):
        if isinstance(self._value, AttributePredicate):
            return self._value.value_for_phase(
                phase
            )
        return self._value


class GreaterThanPredicate(ComparisonPredicate):
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
        True iff the value of the attribute of the phase is greater than
        the value associated with this predicate
        """

        return self.attribute_predicate.value_for_phase(
            phase
        ) > self.value(
            phase
        )


class LessThanPredicate(ComparisonPredicate):
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
        True iff the value of the attribute of the phase is less than
        the value associated with this predicate
        """
        return self.attribute_predicate.value_for_phase(
            phase
        ) < self.value(
            phase
        )


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
        return self.value(
            phase
        ) in self.attribute_predicate.value_for_phase(
            phase
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
        try:
            value = self.value(
                phase
            )
        except AttributeError:
            value = self.value
        return self.attribute_predicate.value_for_phase(
            phase
        ) == value


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
