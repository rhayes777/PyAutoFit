from autofit.database.query_model.condition import NameCondition, AbstractCondition, NullCondition


class NamedQuery(AbstractCondition):
    def __init__(
            self,
            name,
            condition=NullCondition()
    ):
        self.name = name
        self._condition = condition

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

    def __hash__(self):
        return hash(str(self))

    @property
    def tables(self):
        return self.condition.tables
