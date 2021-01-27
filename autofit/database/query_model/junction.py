from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from typing import Set

from .condition import AbstractCondition, Table


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
