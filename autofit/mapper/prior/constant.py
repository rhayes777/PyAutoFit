from autofit.mapper.variable import Variable


class Constant(Variable):
    def __init__(self, value: float):
        """
        Represents a constant value in a model.

        This is equivalent to a prior.

        Note that two different instances of this class with the same value are not equal,
        but are equal once instantiated or in comparison to a float or int.

        Parameters
        ----------
        value
            The constant value.
        """
        super().__init__()
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self):
        return self.id

    def dict(self):
        return {"type": "Constant", "value": self.value}
