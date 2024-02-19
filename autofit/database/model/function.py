import dill

from .model import Object
from ..sqlalchemy_ import sa


class Function(Object):
    """
    An instance, such as a class instance
    """

    __tablename__ = "function"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )
    serialised_function = sa.Column(sa.String)

    @property
    def function(self):
        return dill.loads(self.serialised_function)

    @function.setter
    def function(self, function):
        self.serialised_function = dill.dumps(function)

    __mapper_args__ = {"polymorphic_identity": "instance"}

    @classmethod
    def _from_object(cls, source):
        return cls(function=source)

    def __call__(self):
        return self.function
