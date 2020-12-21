from abc import ABC, abstractmethod

from .instance import *
from .model import *
from .prior import *


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

    def __eq__(self, other):
        return PathQuery(
            *self.queries[:-1],
            self.queries[-1] == other
        )

    @property
    def string(self):
        query_string = self.queries[-1].string
        for query in reversed(
                self.queries[:-1]
        ):
            query_string = f"{query.string} AND id IN ({query_string})"
        return query_string


class EqualityQuery(Query):
    def __init__(self, name_query, value):
        self.name_query = name_query
        self.value = value

    @property
    def tables(self):
        return self.name_query.tables + ["value"]

    @property
    def conditions(self):
        conditions = self.name_query.conditions + [
            f"value = {self.value}"
        ]

        first_table = self.tables[0]
        for table in self.tables[1:]:
            conditions.append(
                f"{table}.id = {first_table}.id"
            )

        return conditions


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
