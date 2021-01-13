import inspect
from abc import ABC, abstractmethod
from numbers import Real
from typing import Optional

from . import condition as c
from .model import Object


class Query(ABC):
    def __init__(
            self,
            parent=None,
            tables=None,
            conditions: Optional[c.ConditionSet] = None
    ):
        self.parent = parent
        self.conditions = conditions or set()
        self.tables = tables or {"object"}

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
        if this.conditions.name_conditions.intersection(
                that.conditions.name_conditions
        ):
            this.conditions.update(
                that.conditions
            )
            return this
        return BranchQuery(
            this, that
        )

    def __comparison(self, symbol, other):
        query = Comparison(
            other,
            symbol
        )
        self.conditions.update(
            query.conditions
        )
        self.tables.update(
            query.tables
        )
        return self

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
        query = Query(
            conditions=c.ConditionSet(c.NameCondition(name)),
            parent=self
        )
        self.conditions.add(
            c.NestedQueryCondition(query)
        )
        return query


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


class Aggregator:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return Query(
            conditions=c.ConditionSet(
                c.NameCondition(name)
            ),
        )

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
