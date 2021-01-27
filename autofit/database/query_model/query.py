import inspect
from numbers import Real

import autofit.database.query_model.condition as c


class NamedQuery(c.AbstractCondition):
    def __init__(
            self,
            name,
            condition: c.AbstractCondition = c.NullCondition()
    ):
        self.name = name
        self._condition = condition

    @property
    def condition(self):
        condition = c.NameCondition(
            self.name
        )
        if self._condition:
            condition &= self._condition
        return condition

    def __repr__(self):
        return self.query

    @property
    def query(self):
        return f"SELECT parent_id FROM {self.condition.tables_string} WHERE {self.condition}"

    def __str__(self):
        return f"id IN ({self.query})"

    def __getattr__(self, item):
        if isinstance(
                self._condition,
                c.NullCondition
        ):
            return NamedQuery(
                self.name,
                NamedQuery(
                    item
                )
            )

        if isinstance(
                self._condition,
                NamedQuery
        ):
            return NamedQuery(
                self.name,
                getattr(
                    self._condition,
                    item
                )
            )

        if isinstance(
                self._condition,
                c.AbstractJunction
        ):
            raise AssertionError(
                "Cannot extend a complex query"
            )

    def __eq__(self, other):
        if isinstance(other, NamedQuery):
            return other.query == self.query

        if isinstance(
                self._condition,
                c.NullCondition
        ):
            return NamedQuery(
                self.name,
                self._make_comparison(other)
            )

        if isinstance(
                self._condition,
                NamedQuery
        ):
            return NamedQuery(
                self.name,
                self._condition == other
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

    def __hash__(self):
        return hash(str(self))

    @property
    def tables(self):
        return self.condition.tables

    def _make_comparison(self, other):
        if isinstance(other, str):
            return c.StringValueCondition(
                "=", other
            )
        if isinstance(other, Real):
            return c.ValueCondition(
                "=", other
            )
        if inspect.isclass(other):
            # if symbol != "=":
            #     raise AssertionError(
            #         "Inequalities to types do not make sense"
            #     )
            return c.TypeCondition(
                other
            )
        raise AssertionError(
            f"Cannot evaluate equality to type {type(other)}"
        )
