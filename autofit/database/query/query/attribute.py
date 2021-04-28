from autofit.database.query import condition as c

from .abstract import AbstractQuery


class AttributeQuery(AbstractQuery):
    @property
    def fit_query(self) -> str:
        """
        The SQL string produced by this query. This is applied directly to the database.
        """
        return f"SELECT id FROM fit WHERE {self.condition}"


class Attribute:
    def __init__(self, attribute: str):
        """
        Some direct attribute of the Fit class

        Parameters
        ----------
        attribute
            The name of that attribute
        """
        self.attribute = attribute

    def _make_query(
            self,
            cls,
            value
    ) -> AttributeQuery:
        """
        Create a query against this attribute

        Parameters
        ----------
        cls
            An AttributeCondition that describes the query
        value
            The value that the attribute is compared to

        Returns
        -------
        A query on ids of the fit table
        """
        return AttributeQuery(
            cls(
                attribute=self.attribute,
                value=value
            )
        )

    def __eq__(self, other) -> AttributeQuery:
        """
        Check whether an attribute, such as a search name, is equal
        to some value
        """
        return self._make_query(
            cls=c.EqualityAttributeCondition,
            value=other
        )

    def contains(self, item: str) -> AttributeQuery:
        """
        Check whether an attribute, such as a search name, contains
        some string
        """
        return self._make_query(
            cls=c.ContainsAttributeCondition,
            value=item
        )


class BooleanAttribute(Attribute, AttributeQuery):
    def __init__(self, attribute):
        super().__init__(attribute)
        super(AttributeQuery, self).__init__(
            c.AttributeCondition(
                attribute
            )
        )

    def __hash__(self):
        return hash(str(self))


class ChildQuery(AttributeQuery):
    def __init__(self, predicate: AbstractQuery):
        super().__init__(
            predicate
        )

    @property
    def condition(self):
        return f"parent_id in ({super().condition})"
