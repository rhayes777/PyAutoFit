import inspect
from abc import ABC, abstractmethod
from numbers import Real

from .model import Object, get_class_path


class Query(ABC):

    @property
    @abstractmethod
    def tables(self):
        pass

    @property
    @abstractmethod
    def conditions(self):
        pass

    @property
    def string(self):
        tables_string = ", ".join(
            self.tables
        )
        conditions_string = " AND ".join(
            self.conditions
        )
        return f"SELECT parent_id FROM {tables_string} WHERE {conditions_string}"


class NameQuery(Query):
    def __init__(self, name):
        self.name = name

    @property
    def tables(self):
        return ["object"]

    @property
    def conditions(self):
        return [f"name = '{self.name}'"]

    def __eq__(self, other):
        return EqualityQuery(
            self,
            other
        )

    def __lt__(self, other):
        return EqualityQuery(
            self,
            other,
            "<"
        )

    def __gt__(self, other):
        return EqualityQuery(
            self,
            other,
            ">"
        )

    def __ge__(self, other):
        return EqualityQuery(
            self,
            other,
            ">="
        )

    def __le__(self, other):
        return EqualityQuery(
            self,
            other,
            "<="
        )

    def __getattr__(self, name):
        return PathQuery(
            self,
            NameQuery(
                name
            )
        )


class PathQuery:
    def __init__(self, *queries):
        self.queries = queries

    def _with_terminating_operation(
            self,
            query
    ):
        return PathQuery(
            *self.queries[:-1],
            query
        )

    @property
    def _terminating_query(self):
        return self.queries[-1]

    def __eq__(self, other):
        return self._with_terminating_operation(
            self._terminating_query == other
        )

    def __gt__(self, other):
        return self._with_terminating_operation(
            self._terminating_query > other
        )

    def __lt__(self, other):
        return self._with_terminating_operation(
            self._terminating_query < other
        )

    def __ge__(self, other):
        return self._with_terminating_operation(
            self._terminating_query >= other
        )

    def __le__(self, other):
        return self._with_terminating_operation(
            self._terminating_query <= other
        )

    def __getattr__(self, name):
        return PathQuery(
            *self.queries,
            NameQuery(
                name
            )
        )

    @property
    def string(self):
        query_string = self.queries[-1].string
        for query in reversed(
                self.queries[:-1]
        ):
            query_string = f"{query.string} AND id IN ({query_string})"
        return query_string


class EqualityQuery(Query, ABC):
    def __new__(cls, name, value, symbol="="):
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
            symbol="="
    ):
        self.name_query = name_query
        self.value = value
        self.symbol = symbol


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
        return self.name_query.tables + [self._table]

    @property
    def conditions(self):
        conditions = self.name_query.conditions + [
            self._condition
        ]

        first_table = self.tables[0]
        for table in self.tables[1:]:
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
