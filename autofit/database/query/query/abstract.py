import copy
from abc import ABC, abstractmethod
from typing import Optional, Set

from autofit.database.query import condition as c
from autofit.database.query.condition import Table


class NotCondition:
    def __init__(self, condition: c.AbstractCondition):
        """
        Prepend the condition with a 'not'

        Parameters
        ----------
        condition
            Some condition such equality to a value
        """
        self._condition = condition

    def __str__(self):
        return f"not ({self._condition})"


class AbstractQuery(c.AbstractCondition, ABC):
    def __init__(
            self,
            condition: Optional[
                c.AbstractCondition
            ] = None
    ):
        """
        A query run to find Fit instances that match given
        criteria

        Parameters
        ----------
        condition
            An optional condition
        """
        self._condition = condition

    @property
    def condition(self):
        return self._condition

    @property
    @abstractmethod
    def fit_query(self) -> str:
        """
        A full query that can be executed against the database to obtain
        fit ids
        """

    def __str__(self):
        return self.fit_query

    @property
    def tables(self) -> Set[Table]:
        return {c.fit_table}

    def __invert__(self):
        """
        Take ~ of this object.

        The object is copied and its condition is prepended
        with a 'not'.
        """
        inverted = copy.deepcopy(self)
        inverted._condition = NotCondition(
            self._condition
        )
        return inverted
