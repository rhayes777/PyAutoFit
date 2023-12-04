from .model import Object
from ..sqlalchemy_ import sa


class Compound(Object):
    __tablename__ = "compound"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
    )
    compound_type = sa.Column(sa.String)

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    __mapper_args__ = {"polymorphic_identity": "compound"}

    def __call__(self):
        return self.cls(self.left(), self.right())

    @classmethod
    def _from_object(cls, compound):
        return Compound(
            compound_type=compound.__class__.__name__,
            children=[
                Object.from_object(compound.left),
                Object.from_object(compound.right),
            ],
            cls=type(compound),
        )
