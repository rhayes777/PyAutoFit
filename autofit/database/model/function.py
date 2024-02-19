import dill

from .model import Object
from ..sqlalchemy_ import sa


class Function(Object):
    """
    A function or callable
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
    def function(self) -> callable:
        """
        The function is stored as a serialised dill string
        """
        return dill.loads(self.serialised_function)

    @function.setter
    def function(self, function):
        self.serialised_function = dill.dumps(function)

    __mapper_args__ = {"polymorphic_identity": "function"}

    @classmethod
    def _from_object(cls, source):
        return cls(function=source)

    def __call__(self):
        return self.function
