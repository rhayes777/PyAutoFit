from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps

from autofit.database.model import get_class_path


class Table:
    def __init__(self, name):
        self.name = name

    @property
    def abbreviation(self):
        return "".join(part[0] for part in self.name.split("_"))

    def __str__(self):
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

    def __gt__(self, other):
        return str(self) > str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __repr__(self):
        return f"<{self.__class__.__name__} {str(self)}>"


class AbstractValueCondition(AbstractCondition, ABC):
    def __init__(self, symbol, value):
        self.value = value
        self.symbol = symbol


class ValueCondition(AbstractValueCondition):
    @property
    def tables(self):
        return {value_table}

    def __str__(self):
        return f"{value_table.abbreviation}.value {self.symbol} {self.value}"


class StringValueCondition(AbstractValueCondition):
    @property
    def tables(self):
        return {string_value_table}

    def __str__(self):
        return f"{string_value_table.abbreviation}.value {self.symbol} '{self.value}'"


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


class NullCondition:
    def __bool__(self):
        return False

    def __and__(self, other):
        return other


def exclude_null(func):
    @wraps(func)
    def wrapper(arg, *conditions):
        conditions = list(filter(
            lambda condition: not isinstance(
                condition,
                NullCondition
            ),
            conditions
        ))
        return func(arg, *conditions)

    return wrapper


class AbstractJunction(AbstractCondition, ABC):
    @exclude_null
    def __new__(cls, *conditions):
        if len(conditions) == 1:
            return conditions[0]
        return object.__new__(cls)

    @exclude_null
    def __init__(
            self,
            *conditions: AbstractCondition
    ):
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
    def tables(self):
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
    def join(self):
        pass

    def __str__(self):
        return f" {self.join} ".join(map(
            str,
            sorted(
                self.conditions
            )
        ))


class And(AbstractJunction):
    @property
    def join(self):
        return "AND"


class Or(AbstractJunction):
    @property
    def join(self):
        return "OR"
