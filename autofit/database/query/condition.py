from abc import ABC, abstractmethod
from typing import Set, Optional

from autofit.database.model import get_class_path


class Table:
    def __init__(
            self,
            name: str,
            abbreviation: Optional[str] = None
    ):
        """
        A table containing some type of object in the database.

        object is parent to all other tables.
        value contains any numeric values.
        string_value contains any string values.

        Parameters
        ----------
        name
            The name of the table
        """
        self.name = name
        self._abbreviation = abbreviation

    @property
    def abbreviation(self) -> str:
        """
        A one letter abbreviation used as an alias for this table
        """
        if self._abbreviation is not None:
            return self._abbreviation
        return "".join(part[0] for part in self.name.split("_"))

    def __str__(self):
        """
        Describes the table in the FROM or JOIN statement
        """
        return f"{self.name} AS {self.abbreviation}"

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name

    def __hash__(self):
        return hash(self.name)


object_table = Table("object")
value_table = Table("value")
string_value_table = Table("string_value")
fit_table = Table("fit")
info_table = Table("info", "info")
none_table = Table("none")


class AbstractCondition(ABC):
    @property
    @abstractmethod
    def tables(self) -> Set[Table]:
        """
        The set of tables this condition applies to
        """

    @abstractmethod
    def __str__(self):
        """
        The condition written as SQL
        """

    def __and__(
            self,
            other:
            "AbstractCondition"
    ):
        """
        Combine this and another query with an AND statement.
        
        Simplification is applied so that the query will execute as fast as possible.
        """
        from .junction import And
        return And(
            self,
            other
        )

    def __or__(
            self,
            other: "AbstractCondition"
    ):
        """
        Combine this and another query with an AND statement.

        Simplification is applied so that the query will execute as fast as possible.
        """
        from .junction import Or
        return Or(
            self,
            other
        )

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __repr__(self):
        return f"<{self.__class__.__name__} {str(self)}>"


class NoneCondition(AbstractCondition):

    @property
    def tables(self) -> Set[Table]:
        return {none_table}

    def __str__(self):
        return "1 = 1"


class AbstractValueCondition(AbstractCondition, ABC):
    def __init__(
            self,
            symbol: str,
            value
    ):
        """
        A condition which compares the named column to a value

        Parameters
        ----------
        symbol
            =, <=, >=, < or >
        value
            A string, int or float
        """
        self.value = value
        self.symbol = symbol


class ValueCondition(AbstractValueCondition):
    @property
    def tables(self) -> Set[Table]:
        """
        Tables included in the condition string
        """
        return {value_table}

    def __str__(self) -> str:
        """
        The condition in SQL
        """
        return f"{value_table.abbreviation}.value {self.symbol} {self.value}"


class StringValueCondition(AbstractValueCondition):
    @property
    def tables(self):
        """
        Tables included in the condition string
        """
        return {string_value_table}

    def __str__(self):
        """
        The condition in SQL
        """
        return f"{string_value_table.abbreviation}.value {self.symbol} '{self.value}'"


class NameCondition(AbstractCondition):
    def __init__(self, name: str):
        """
        Condition used to match the name of an object. e.g. galaxy, lens, brightness.

        Parameters
        ----------
        name
            The name of an attribute in the model
        """
        self.name = name

    @property
    def tables(self):
        """
        Tables included in the condition string
        """
        return {object_table}

    def __str__(self):
        """
        The condition in SQL
        """
        return f"{object_table.abbreviation}.name = '{self.name}'"


class TypeCondition(AbstractCondition):
    def __init__(self, cls: type):
        """
        Condition used to match the type of an object. e.g. SersicLightProfile

        Parameters
        ----------
        cls
            The type
        """
        self.cls = cls

    @property
    def tables(self):
        """
        Tables included in the condition string
        """
        return {object_table}

    def __str__(self):
        """
        The condition in SQL
        """
        return f"{object_table.abbreviation}.class_path = '{self.class_path}'"

    @property
    def class_path(self) -> str:
        """
        The full import path of the type
        """
        return get_class_path(
            self.cls
        )


class AttributeCondition(AbstractCondition, ABC):
    def __init__(self, attribute, value):
        self.attribute = attribute
        self._value = value

    @property
    def tables(self) -> Set[Table]:
        return {fit_table}

    @abstractmethod
    def __str__(self):
        pass


class EqualityAttributeCondition(AttributeCondition):
    @property
    def value(self):
        if isinstance(
                self._value,
                str
        ):
            return f"'{self._value}'"
        return self._value

    def __str__(self):
        return f"{self.attribute} = {self.value}"


class ContainsAttributeCondition(AttributeCondition):
    def __str__(self):
        return f"{self.attribute} LIKE '%{self._value}%'"


class InAttributeCondition(AttributeCondition):
    def __str__(self):
        return f"'{self._value}' LIKE '%' || {self.attribute} || '%'"


class AttributeCondition(AbstractCondition):
    def __init__(self, attribute):
        self.attribute = attribute

    @property
    def tables(self) -> Set[Table]:
        return {fit_table}

    def __str__(self):
        return self.attribute
