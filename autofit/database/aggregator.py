import inspect
from abc import ABC, abstractmethod
from numbers import Real
from typing import Set

from .model import Object, get_class_path


class Query(ABC):
    def __init__(
            self,
            parent=None
    ):
        self.parent = parent
        self.children = []
        if self.parent is not None:
            self.parent.children = [self]

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def tables(self) -> Set[str]:
        pass

    @property
    @abstractmethod
    def conditions(self):
        pass

    @property
    def _string(self):
        tables_string = ", ".join(
            sorted(self.tables)
        )
        conditions_string = " AND ".join(
            sorted(self.conditions)
        )

        string = f"SELECT parent_id FROM {tables_string} WHERE {conditions_string}"

        if len(self.children) > 0:
            children_strings = " AND ".join(
                f"id IN ({child._string})"
                for child
                in self.children
            )
            string = f"{string} AND {children_strings}"

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
            this.children.extend(
                that.children
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


class NameQuery(Query):
    def __init__(
            self,
            name,
            parent=None
    ):
        super().__init__(
            parent=parent
        )
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def tables(self):
        return {"object"}

    @property
    def conditions(self):
        return [f"name = '{self.name}'"]

    def __comparison(self, symbol, other):
        return EqualityQuery(
            self,
            other,
            symbol,
            parent=self.parent
        )

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
        return NameQuery(
            name,
            parent=self
        )


class ConjunctionQuery(Query):
    def __init__(
            self,
            *child_queries,
            parent
    ):
        super().__init__(parent)
        self.child_queries = child_queries

    @property
    def name(self):
        return self.child_queries[0].name

    @property
    def tables(self):
        return {
            table
            for query in self.child_queries
            for table in query.tables
        }

    @property
    def conditions(self):
        return {
            condition
            for query in self.child_queries
            for condition in query.conditions
        }


class EqualityQuery(Query, ABC):
    def __new__(
            cls,
            name,
            value,
            symbol="=",
            *,
            parent
    ):
        if isinstance(value, str):
            return object.__new__(StringEqualityQuery)
        if isinstance(value, Real):
            return object.__new__(ValueEqualityQuery)
        if inspect.isclass(value):
            if symbol != "=":
                raise AssertionError(
                    "Inequalities to types do not make sense"
                )
            return object.__new__(TypeEqualityQuery)
        raise AssertionError(
            f"Cannot evaluate equality to type {type(value)}"
        )

    def __init__(
            self,
            name_query,
            value,
            symbol="=",
            *,
            parent=None
    ):
        super().__init__(parent)
        self.name_query = name_query
        self.value = value
        self.symbol = symbol

    @property
    def name(self):
        return self.name_query.name


class RegularEqualityQuery(EqualityQuery, ABC):
    @property
    @abstractmethod
    def _table(self):
        pass

    @property
    @abstractmethod
    def _condition(self):
        pass

    @property
    def tables(self):
        return {*self.name_query.tables, self._table}

    @property
    def conditions(self):
        conditions = self.name_query.conditions + [
            self._condition
        ]

        tables = sorted(self.tables)
        first_table = tables[0]
        for table in tables[1:]:
            conditions.append(
                f"{table}.id = {first_table}.id"
            )

        return conditions


class StringEqualityQuery(RegularEqualityQuery):
    @property
    def _table(self):
        return "string_value"

    @property
    def _condition(self):
        return f"value {self.symbol} '{self.value}'"


class ValueEqualityQuery(RegularEqualityQuery):
    @property
    def _table(self):
        return "value"

    @property
    def _condition(self):
        return f"value {self.symbol} {self.value}"


class TypeEqualityQuery(EqualityQuery):
    @property
    def tables(self):
        return ["object"]

    @property
    def conditions(self):
        return self.name_query.conditions + [
            f"class_path = '{get_class_path(self.value)}'"
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
