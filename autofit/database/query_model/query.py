from autofit.database.query_model.condition import NameCondition, And, AbstractCondition


class NamedQuery(AbstractCondition):
    def __init__(
            self,
            name,
            condition=None
    ):
        self.name = name
        self.condition = condition

    def __and__(self, other):
        return NamedQuery(
            self.name,
            And(
                self.condition,
                other.condition
            )
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        condition = NameCondition(
            self.name
        )

        if self.condition is not None:
            condition &= self.condition
        return f"SELECT parent_id FROM {condition.tables_string} WHERE {condition}"

    def __eq__(self, other):
        return str(self) == str(other)

    @property
    def tables(self):
        return self.condition.tables
