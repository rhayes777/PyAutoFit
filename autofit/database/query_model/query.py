import inspect
from numbers import Real
from typing import Optional

import autofit.database.query_model.condition as c


def _make_comparison(symbol, other):
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
            name,
            condition: Optional[c.AbstractCondition] = None
    ):
        self.name = name
        self.other_condition = condition

    @property
    def condition(self):
        condition = c.NameCondition(
            self.name
        )
        if self.other_condition:
            condition &= self.other_condition
        return condition

    def __repr__(self):
        return self.query

    @property
    def tables_string(self):
        tables = sorted(self.condition.tables)
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
    def query(self):
        return f"SELECT parent_id FROM {self.tables_string} WHERE {self.condition}"

    def __str__(self):
        return f"o.id IN ({self.query})"

    def __getattr__(self, item):
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

        if isinstance(
                self.other_condition,
                c.AbstractJunction
        ):
            raise AssertionError(
                "Cannot extend a complex query"
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

    def _recursive_comparison(self, symbol, other):
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
                c.AbstractJunction
        ):
            raise AssertionError(
                "Cannot compare a complex query"
            )

        raise AssertionError(
            f"Cannot evaluate equality to type {type(other)}"
        )
