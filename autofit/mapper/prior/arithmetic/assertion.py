from abc import ABC
from typing import Optional, Dict

from autofit.mapper.prior.arithmetic.compound import CompoundPrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel


class Assertion:
    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        assertion_type = d.pop("assertion_type")
        for subclass in cls.descendants():
            if subclass.__name__ == assertion_type:
                return subclass.from_dict(d)
        raise ValueError(f"Assertion type {assertion_type} not recognised")

    @classmethod
    def descendants(cls):
        subclasses = cls.__subclasses__()
        descendants = set(cls.__subclasses__())

        for child in subclasses:
            descendants.update(child.descendants())
        return descendants


class ComparisonAssertion(CompoundPrior, Assertion, ABC):
    def __init__(self, lower, greater, name=""):
        super().__init__(lower, greater)
        self._name = name

    def dict(self) -> dict:
        from autofit import ModelObject

        return {
            "type": "assertion",
            "assertion_type": self.__class__.__name__,
            "lower": self._left.dict()
            if isinstance(self._left, ModelObject)
            else self._left,
            "greater": self._right.dict()
            if isinstance(self._right, ModelObject)
            else self._right,
        }

    @classmethod
    def from_dict(
        cls,
        d,
        reference: Optional[Dict[str, str]] = None,
        loaded_ids: Optional[dict] = None,
    ):
        from autofit import ModelObject

        return cls(
            ModelObject.from_dict(d["lower"]),
            ModelObject.from_dict(d["greater"]),
        )

    def __gt__(self, other):
        return CompoundAssertion(self, self._left > other)

    def __lt__(self, other):
        return CompoundAssertion(self, self._right < other)

    def __ge__(self, other):
        return CompoundAssertion(self, self._left >= other)

    def __le__(self, other):
        return CompoundAssertion(self, self._right <= other)


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
        return self.left_for_arguments(arguments) <= self.right_for_arguments(arguments)


class CompoundAssertion(AbstractPriorModel, Assertion):
    def __init__(self, assertion_1, assertion_2, name=""):
        super().__init__()
        self.assertion_1 = assertion_1
        self.assertion_2 = assertion_2
        self._name = name

    def _instance_for_arguments(self, arguments):
        return self.assertion_1.instance_for_arguments(
            arguments,
        ) and self.assertion_2.instance_for_arguments(
            arguments,
        )

    def dict(self) -> dict:
        return {
            "type": "assertion",
            "assertion_type": self.__class__.__name__,
            "assertion_1": self.assertion_1.dict(),
            "assertion_2": self.assertion_2.dict(),
        }


def unwrap(obj):
    try:
        return obj._value
    except AttributeError:
        return obj
