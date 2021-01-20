from abc import ABC, abstractmethod


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


class And(AbstractCondition):
    def __init__(
            self,
            condition_1: AbstractCondition,
            condition_2: AbstractCondition
    ):
        self.condition_1 = condition_1
        self.condition_2 = condition_2

    @property
    def tables(self):
        return {
            *self.condition_1.tables,
            *self.condition_2.tables
        }

    def __str__(self):
        return f"{self.condition_1} AND {self.condition_2}"


class NamedQuery:
    def __init__(
            self,
            name,
            condition=None
    ):
        self.name = name
        self.condition = NameCondition(
            self.name
        )

        if condition is not None:
            self.condition &= condition

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"SELECT parent_id FROM {self.condition.tables_string} WHERE {self.condition}"

    def __eq__(self, other):
        return str(self) == str(other)
