from typing import Set

from autofit.database.query.condition import AbstractCondition, Table, info_table
from autofit.database.query.query import AbstractQuery


class InfoQueryCondition(AbstractCondition):
    def __init__(self, key: str, value: str):
        """
        Query some item stored in the fit info dictionary

        Parameters
        ----------
        key
        value
        """
        self.key = key
        self.value = value

    @property
    def tables(self) -> Set[Table]:
        return {info_table}

    def __str__(self):
        return f"key = '{self.key}' AND value = '{self.value}'"


class InfoQuery(AbstractQuery):
    def __init__(self, key, value):
        super().__init__(
            InfoQueryCondition(
                key=key,
                value=value
            )
        )

    @property
    def fit_query(self) -> str:
        return f"SELECT fit_id FROM info WHERE {self.condition}"


class InfoField:
    def __init__(self, key):
        self.key = key

    def __eq__(self, other):
        return InfoQuery(
            self.key,
            other
        )


class AnonymousInfo:
    def __getitem__(self, item):
        return InfoField(item)
