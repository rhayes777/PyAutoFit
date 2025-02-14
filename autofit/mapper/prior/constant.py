from numbers import Number

from autofit.mapper.model_object import ModelObject


class Constant(ModelObject, Number):
    def __init__(self, value: float, id_=None):
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
        super().__init__(
            id_=id_,
        )
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return self.value == other
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.id)

    def dict(self):
        return {"type": "Constant", "value": self.value}

    def __add__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __mul__(self, other):
        return self.value * other

    def __truediv__(self, other):
        return self.value / other

    def __pow__(self, other):
        return self.value**other

    def __radd__(self, other):
        return other + self.value

    def __rsub__(self, other):
        return other - self.value

    def __rmul__(self, other):
        return other * self.value

    def __rtruediv__(self, other):
        return other / self.value

    def __rpow__(self, other):
        return other**self.value

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other
