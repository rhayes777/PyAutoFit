import inspect
from numbers import Real
from typing import Optional, Set

import autofit.database.query_model.condition as c
from autofit.database.query_model.junction import AbstractJunction


def _make_comparison(
        symbol: str,
        other
) -> c.AbstractCondition:
    """
    Create the appropriate comparison class, depending on the type of other.

    Parameters
    ----------
    symbol
        The symbol by which the condition will compare
    other
        Some object to be compared to entries in the database

    Returns
    -------
    A condition
    """
    if isinstance(other, str):
        return c.StringValueCondition(
            symbol, other
        )
    if isinstance(other, Real):
        return c.ValueCondition(
            symbol, other
        )
    if inspect.isclass(other):
        if symbol != "=":
            raise AssertionError(
                "Inequalities to types do not make sense"
            )
        return c.TypeCondition(
            other
        )
    raise AssertionError(
        f"Cannot evaluate equality to type {type(other)}"
    )


class NamedQuery(c.AbstractCondition):
    def __init__(
            self,
            name: str,
            condition: Optional[c.AbstractCondition] = None
    ):
        """
        An object which can be converted into SQL and used to query the database.

        Model instances are flat packed into the database using the name of an
        attribute and a foreign key of the parent's id to express the original
        parent - attribute relationships.

        Queries can be constructed on these models.

        e.g.
        aggregator.galaxies.lens == al.lp.SersicLightProfile

        This class is responsible for the construction of queries from this syntax.
        The name is the name in the path, e.g. galaxies.

        Parameters
        ----------
        name
            The name of an attribute that is stored in the database
        condition
            A child condition combined with the name to produce a query
        """
        self.name = name
        self.other_condition = condition

    @property
    def condition(self) -> c.AbstractCondition:
        """
        The combined condition that forms the WHERE statement of the query.

        This is at least a check on the name in the object table, but may
        include subqueries of arbitrary complexity expressed by 'other_condition'
        """
        condition = c.NameCondition(
            self.name
        )
        if self.other_condition:
            condition &= self.other_condition
        return condition

    @property
    def tables(self) -> Set[c.Table]:
        """
        The set of tables used in the FROM component of the query
        """
        return self.condition.tables

    def __repr__(self):
        return self.query

    @property
    def tables_string(self) -> str:
        """
        A string found in the FROM component. Includes aliasing and join
        statements if there is more than one table.

        Currently it is assumed that only the object table and up to one
        further table are required to execute any given query.
        """
        tables = sorted(self.tables)
        first = tables[0]

        string = str(first)

        if len(tables) == 2:
            second = tables[1]
            string = f"{string} JOIN {second} ON {first.abbreviation}.id = {second.abbreviation}.id"
        if len(tables) > 2:
            raise AssertionError(
                "Currently maximum of 2 tables supported"
            )

        return string

    @property
    def query(self) -> str:
        """
        The SQL string produced by this query. This is applied directly to the database.
        """
        return f"SELECT parent_id FROM {self.tables_string} WHERE {self.condition}"

    def __str__(self):
        return f"o.id IN ({self.query})"

    def __getattr__(self, item: str):
        """
        Used to extend the query.

        Supports the syntax:
        galaxies.lens.intensity

        Child queries are recursively searched until one with a None child is found.
        A new query is constructed to replace this None, extending the linked list
        of Named queries.

        If any child query is not a Named query then an exception is thrown as this
        does not make sense.

        For example, (one.two.three == 1).four does not make sense.

        Parameters
        ----------
        item
            The name of an attribute

        Returns
        -------
        A newly created query that is the same as this query but with an additional
        NamedQuery added on the end for the new attribute.
        """
        if self.other_condition is None:
            return NamedQuery(
                self.name,
                NamedQuery(
                    item
                )
            )

        if isinstance(
                self.other_condition,
                NamedQuery
        ):
            return NamedQuery(
                self.name,
                getattr(
                    self.other_condition,
                    item
                )
            )

        raise AssertionError(
            "Can only extend a simple path query"
        )

    def __eq__(self, other):
        if isinstance(other, NamedQuery):
            return other.query == self.query

        return self._recursive_comparison(
            "=",
            other
        )

    def __gt__(self, other):
        if isinstance(other, c.AbstractCondition):
            return super().__gt__(other)
        return self._recursive_comparison(
            ">",
            other
        )

    def __ge__(self, other):
        return self._recursive_comparison(
            ">=",
            other
        )

    def __lt__(self, other):
        if isinstance(other, c.AbstractCondition):
            return super().__lt__(other)
        return self._recursive_comparison(
            "<",
            other
        )

    def __le__(self, other):
        return self._recursive_comparison(
            "<=",
            other
        )

    def __hash__(self):
        return hash(str(self))

    def _recursive_comparison(
            self,
            symbol: str,
            other
    ) -> "NamedQuery":
        """
        Create a new NamedQuery, recursing through each sub NamedQuery until
        one without a child query is found. The final NamedQuery is given a
        child query.

        This supports the syntax:
        aggregator.galaxies.centre.intensity >= 1

        The final query added on the end is the comparison query ">= 1"

        Parameters
        ----------
        symbol
            =, >=, <=, >, <
        other
            An object to compare to values in the database.

        Returns
        -------
        A new NamedQuery with the comparison query added on the end
        """
        if self.other_condition is None:
            return NamedQuery(
                self.name,
                _make_comparison(
                    symbol,
                    other
                )
            )

        if isinstance(
                self.other_condition,
                NamedQuery
        ):
            return NamedQuery(
                self.name,
                self.other_condition == other
            )

        if isinstance(
                self._condition,
                AbstractJunction
        ):
            raise AssertionError(
                "Cannot compare a complex query"
            )

        raise AssertionError(
            f"Cannot evaluate equality to type {type(other)}"
        )
