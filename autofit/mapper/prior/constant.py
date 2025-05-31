from numbers import Number

from autofit.mapper.model_object import ModelObject


class Constant(float, ModelObject):
    def __new__(cls, value: float, id_=None):
        obj = super().__new__(cls, value)
        return obj

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
        if isinstance(value, Constant):
            value = value.value

        ModelObject.__init__(self, id_=id_)
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, ModelObject):
            return ModelObject.__eq__(self, other)
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, ModelObject):
            return ModelObject.__ne__(self, other)
        return self.value != other

    def __hash__(self):
        return hash(self.id)

    def dict(self):
        return {"type": "Constant", "value": self.value}
