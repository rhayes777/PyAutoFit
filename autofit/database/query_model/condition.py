from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from typing import Set

from autofit.database.model import get_class_path


class Table:
    def __init__(self, name: str):
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

    @property
    def abbreviation(self) -> str:
        """
        A one letter abbreviation used as an alias for this table
        """
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


class AbstractCondition(ABC):
    @abstractmethod
    def __str__(self):
        """
        The condition written as SQL
        """

    def __and__(
            self,
            other:
            "AbstractCondition"
    ) -> "And":
        """
        Combine this and another query with an AND statement.
        
        Simplification is applied so that the query will execute as fast as possible.
        """
        return And(
            self,
            other
        )

    def __or__(
            self,
            other: "AbstractCondition"
    ) -> "Or":
        """
        Combine this and another query with an AND statement.

        Simplification is applied so that the query will execute as fast as possible.
        """
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


def exclude_none(func):
    """
    Decorator that filters None from an argument list of conditions
    """
    @wraps(func)
    def wrapper(arg, *conditions):
        conditions = list(filter(
            None,
            conditions
        ))
        return func(arg, *conditions)

    return wrapper


class AbstractJunction(AbstractCondition, ABC):
    @exclude_none
    def __new__(
            cls,
            *conditions: AbstractCondition
    ):
        """
        If only a single extant condition is passed in then that
        condition should simply be returned.
        """
        if len(conditions) == 1:
            return conditions[0]
        return object.__new__(cls)

    @exclude_none
    def __init__(
            self,
            *conditions: AbstractCondition
    ):
        """
        A combination of two or more conditions. A set of rules allow
        the query to be expressed in the simplest terms possible

        Any subjunctions of the same type are unwrapped.
        i.e. And(A, B) & And(C, D) -> And(A, B, C, D)

        Conditions are a set so duplication is removed.
        i.e. And(A, B, A) -> And(A, B)

        NamedQuery child conditions are matched by their name creating
        one NameQuery for each name and applying the junction type to
        their conditions.
        i.e. And(Named('name', A), Named('name', B)) -> Named('name', And(A, B))

        Parameters
        ----------
        conditions
            A list of SQL conditions
        """
        from .query import NamedQuery

        self.conditions = set()

        named_query_dict = defaultdict(set)

        def add_conditions(conditions_):
            for condition in conditions_:
                if isinstance(
                        condition,
                        self.__class__
                ):
                    add_conditions(condition)
                elif isinstance(
                        condition,
                        NamedQuery
                ):
                    named_query_dict[
                        condition.name
                    ].add(
                        condition
                    )
                else:
                    self.conditions.add(condition)

        add_conditions(conditions)

        for name, queries in named_query_dict.items():
            # noinspection PyTypeChecker
            self.conditions.add(
                NamedQuery(
                    name,
                    self.__class__(
                        *[
                            query.other_condition
                            for query
                            in queries
                        ]
                    )
                )
            )

    def __iter__(self):
        return iter(sorted(self.conditions))

    @property
    def tables(self) -> Set[Table]:
        """
        Combines the tables of all subqueries which are not
        named queries
        """
        from .query import NamedQuery
        return {
            table
            for condition
            in self.conditions
            if not isinstance(
                condition,
                NamedQuery
            )
            for table
            in condition.tables
        }

    @property
    @abstractmethod
    def join(self) -> str:
        """
        SQL string used to conjoin queries
        """

    def __str__(self) -> str:
        """
        SQL string expressing combined query
        """
        return f" {self.join} ".join(map(
            str,
            sorted(
                self.conditions
            )
        ))


class And(AbstractJunction):
    @property
    def join(self) -> str:
        return "AND"


class Or(AbstractJunction):
    @property
    def join(self):
        return "OR"
