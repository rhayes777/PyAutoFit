from autofit.database.query_model.condition import NameCondition, And, AbstractCondition


class NamedQuery(AbstractCondition):
    def __init__(
            self,
            name,
            condition=None
    ):
        self.name = name
        self._condition = condition

    def __and__(self, other):
        if self.name == other.name:
            return NamedQuery(
                self.name,
                And(
                    self._condition,
                    other._condition
                )
            )

    @property
    def condition(self):
        condition = NameCondition(
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

    def __eq__(self, other):
        try:
            return str(other) == str(self) or other == self.query or other.query == self.query
        except AttributeError:
            return False

    @property
    def tables(self):
        return self.condition.tables
