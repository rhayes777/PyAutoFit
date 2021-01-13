import inspect
from abc import ABC, abstractmethod
from numbers import Real
from typing import List

from .model import Object, get_class_path


class Condition(ABC):
    @abstractmethod
    def __str__(self):
        pass

    def __hash__(self):
        return hash(str(self))


class Query(ABC):
    def __init__(
            self,
            parent=None,
            tables=None
    ):
        self.parent = parent
        self.child_conditions = []
        self.tables = tables or set()

    @property
    def conditions(self):
        return self.child_conditions

    @property
    @abstractmethod
    def name(self):
        pass

    def __str__(self):
        return self._string

    @property
    def _string(self):
        tables_string = ", ".join(
            sorted(self.tables)
        )
        conditions_string = " AND ".join(
            sorted(map(str, self.conditions))
        )

        string = f"SELECT parent_id FROM {tables_string} WHERE {conditions_string}"

        # if len(self.children) > 0:
        #     children_strings = " AND ".join(
        #         f"id IN ({child._string})"
        #         for child
        #         in self.children
        #     )
        #     string = f"{string} AND {children_strings}"

        return string

    @property
    def top_level(self):
        if self.parent is not None:
            return self.parent.top_level
        return self

    @property
    def string(self):
        return self.top_level._string

    def __and__(self, other):
        this = self.top_level
        that = other.top_level
        if this.name == that.name:
            this.child_conditions.extend(
                that.child_conditions
            )
            return this
        return BranchQuery(
            this, that
        )


class BranchQuery:
    def __init__(self, *child_queries):
        self.child_queries = child_queries

    @property
    def string(self):
        subqueries = [
            f"({query.string}) as t{number}"
            for number, query
            in enumerate(
                self.child_queries
            )
        ]
        conditions = [
            f"t0.parent_id = t{number}.parent_id"
            for number
            in range(1, len(
                self.child_queries
            ))
        ]
        return f"SELECT t0.parent_id FROM {', '.join(subqueries)} WHERE {'AND'.join(conditions)}"


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

    def __hash__(self):
        return self.cls

    def __str__(self):
        return f"class_path = '{get_class_path(self.cls)}'"


class NameQuery(Query):
    def __init__(
            self,
            name,
            parent=None
    ):
        super().__init__(
            parent=parent,
            tables={"object"}
        )
        self._name = name

    @property
    def name(self):
        return self._name

    def __comparison(self, symbol, other):
        query = ComparisonQuery(
            other,
            symbol
        )
        self.child_conditions.extend(
            query.conditions
        )
        self.tables.update(
            query.tables
        )
        return self

    @property
    def conditions(self):
        return super().conditions + [
            NameCondition(self.name)
        ]

    def __eq__(self, other):
        return self.__comparison("=", other)

    def __lt__(self, other):
        return self.__comparison("<", other)

    def __gt__(self, other):
        return self.__comparison(">", other)

    def __ge__(self, other):
        return self.__comparison(">=", other)

    def __le__(self, other):
        return self.__comparison("<=", other)

    def __getattr__(self, name):
        query = NameQuery(
            name,
            parent=self
        )
        self.child_conditions.append(
            NestedQueryCondition(query)
        )
        return query


def wrap_string(value):
    if isinstance(value, str):
        return f"'{value}'"
    return value


class ComparisonCondition(Condition):
    def __init__(self, column, value, symbol):
        self.column = column
        self.value = wrap_string(value)
        self.symbol = symbol

    def __str__(self):
        return f"{self.column} {self.symbol} {self.value}"


class ComparisonQuery:
    def __new__(
            cls,
            value,
            symbol="=",
    ):
        if isinstance(value, str):
            return object.__new__(StringComparisonQuery)
        if isinstance(value, Real):
            return object.__new__(ValueComparisonQuery)
        if inspect.isclass(value):
            if symbol != "=":
                raise AssertionError(
                    "Inequalities to types do not make sense"
                )
            return object.__new__(TypeComparisonQuery)
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


class RegularComparisonQuery(ComparisonQuery, ABC):
    @property
    def conditions(self):
        return [
            JoinCondition(
                "object",
                *self.tables
            ),
            ComparisonCondition(
                column="value",
                value=self.value,
                symbol=self.symbol
            )
        ]


class StringComparisonQuery(RegularComparisonQuery):
    @property
    def tables(self):
        return ["string_value"]


class ValueComparisonQuery(RegularComparisonQuery):
    @property
    def tables(self):
        return ["value"]


class TypeComparisonQuery(ComparisonQuery):
    @property
    def tables(self):
        return []

    @property
    def conditions(self) -> List[Condition]:
        return [
            ClassPathCondition(self.value)
        ]


class Aggregator:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return NameQuery(name)

    def filter(self, predicate):
        objects_ids = {
            row[0]
            for row
            in self.session.execute(
                predicate.string
            )
        }
        return self.session.query(
            Object
        ).filter(
            Object.id.in_(
                objects_ids
            )
        ).all()
