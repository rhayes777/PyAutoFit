from .instance import *
from .model import *
from .prior import *


class NameQuery:
    def __init__(self, name):
        self.name = name

    @property
    def tables(self):
        return ["object"]

    @property
    def conditions(self):
        return [f"name = '{self.name}'"]

    @property
    def string(self):
        tables_string = ", ".join(
            self.tables
        )
        conditions_string = " AND ".join(
            self.conditions
        )
        return f"SELECT parent_id FROM {tables_string} WHERE {conditions_string}"

    def __eq__(self, other):
        return EqualityQuery(
            self.name,
            other
        )


class EqualityQuery(NameQuery):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value

    @property
    def tables(self):
        return super().tables + ["value"]

    @property
    def conditions(self):
        conditions = super().conditions + [
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
