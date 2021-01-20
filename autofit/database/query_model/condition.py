from abc import ABC, abstractmethod

from autofit.database import get_class_path


class Table:
    def __init__(self, name):
        self.name = name

    @property
    def abbreviation(self):
        return self.name[0]

    def __str__(self):
        return f"{self.name} AS {self.abbreviation}"


object_table = Table("object")
value_table = Table("value")


class AbstractCondition(ABC):
    @property
    @abstractmethod
    def tables(self):
        pass

    @property
    def tables_string(self):
        return ", ".join(sorted(map(str, self.tables)))

    @abstractmethod
    def __str__(self):
        pass

    def __and__(self, other):
        return And(
            self,
            other
        )

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class ValueCondition(AbstractCondition):
    def __init__(self, symbol, value):
        self.value = value
        self.symbol = symbol

    @property
    def tables(self):
        return {value_table}

    def __str__(self):
        return f"{value_table.abbreviation}.value {self.symbol} {self.value}"


class NameCondition(AbstractCondition):
    def __init__(self, name):
        self.name = name

    @property
    def tables(self):
        return {object_table}

    def __str__(self):
        return f"{object_table.abbreviation}.name = '{self.name}'"


class TypeCondition(AbstractCondition):
    def __init__(self, cls):
        self.cls = cls

    @property
    def tables(self):
        return {object_table}

    @property
    def class_path(self):
        return get_class_path(
            self.cls
        )

    def __str__(self):
        return f"{object_table.abbreviation}.class_path = '{self.class_path}'"


class And(AbstractCondition):
    def __init__(
            self,
            condition_1: AbstractCondition,
            condition_2: AbstractCondition
    ):
        self.condition_1 = condition_1
        self.condition_2 = condition_2

    def __bool__(self):
        return bool(self.condition_1) and bool(self.condition_2)

    def __new__(cls, condition_1, condition_2):
        if condition_1 == condition_2:
            return condition_1
        if condition_1 is None:
            return condition_2
        if condition_2 is None:
            return condition_1
        if condition_1 is None and condition_2 is None:
            return None
        return object.__new__(And)

    @property
    def tables(self):
        tables = set()
        for condition in (
                self.condition_1,
                self.condition_2
        ):
            if condition:
                tables.update(
                    condition.tables
                )
        return tables

    def __str__(self):
        if self.condition_1 and self.condition_2:
            return f"{self.condition_1} AND {self.condition_2}"
        if self.condition_1:
            return str(self.condition_1)
        if self.condition_2:
            return str(self.condition_2)
