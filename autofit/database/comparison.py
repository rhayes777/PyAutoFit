import inspect
from abc import ABC, abstractmethod
from numbers import Real

from . import condition as c


class Comparison:
    def __new__(
            cls,
            value,
            symbol="=",
    ):
        if isinstance(value, str):
            return object.__new__(StringComparison)
        if isinstance(value, Real):
            return object.__new__(ValueComparison)
        if inspect.isclass(value):
            if symbol != "=":
                raise AssertionError(
                    "Inequalities to types do not make sense"
                )
            return object.__new__(TypeComparison)
        raise AssertionError(
            f"Cannot evaluate equality to type {type(value)}"
        )

    def __init__(
            self,
            value,
            symbol="=",
    ):
        self.value = value
        self.symbol = symbol

    @property
    @abstractmethod
    def tables(self):
        pass

    @property
    @abstractmethod
    def conditions(self):
        pass


class RegularComparison(Comparison, ABC):
    @property
    def conditions(self):
        return {
            c.JoinCondition(
                "object",
                *self.tables
            ),
            c.ComparisonCondition(
                column="value",
                value=self.value,
                symbol=self.symbol
            )
        }


class StringComparison(RegularComparison):
    @property
    def tables(self):
        return {"string_value"}


class ValueComparison(RegularComparison):
    @property
    def tables(self):
        return {"value"}


class TypeComparison(Comparison):
    @property
    def tables(self):
        return {}

    @property
    def conditions(self) -> c.ConditionSet:
        return c.ConditionSet(
            c.ClassPathCondition(self.value)
        )
