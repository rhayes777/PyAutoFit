from abc import ABC, abstractmethod
from typing import Set

from .model import get_class_path


def wrap_string(value):
    if isinstance(value, str):
        return f"'{value}'"
    return value


class Condition(ABC):
    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __gt__(self, other):
        return str(self) > other

    def __lt__(self, other):
        return str(self) < other


class ConditionSet:
    def __init__(self, name):
        assert isinstance(name, str)

        self.name = name

        self.conditions = {
            NameCondition(name)
        }

    def __getattr__(self, item):
        return getattr(
            self.conditions,
            item
        )

    def __iter__(self):
        return iter(sorted(self.conditions))


class NestedQueryCondition(Condition):
    def __init__(self, query):
        self.query = query

    def __str__(self):
        return f"id IN ({str(self.query)})"


class NameCondition(Condition):
    def __str__(self):
        return f"name = '{self.name}'"

    def __init__(self, name):
        self.name = name


class JoinCondition(Condition):
    def __str__(self):
        conditions = []
        for table in sorted(self.other_tables):
            conditions.append(
                f"{table}.id = {self.primary_table}.id"
            )
        return " AND ".join(conditions)

    def __init__(self, primary_table, *other_tables):
        self.primary_table = primary_table
        self.other_tables = other_tables


class ClassPathCondition(Condition):
    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return f"class_path = '{get_class_path(self.cls)}'"


class ComparisonCondition(Condition):
    def __init__(self, column, value, symbol):
        self.column = column
        self.value = wrap_string(value)
        self.symbol = symbol

    def __str__(self):
        return f"{self.column} {self.symbol} {self.value}"
