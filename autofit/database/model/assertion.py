from .model import Object
from ..sqlalchemy_ import sa


class Assertion(Object):
    __tablename__ = "assertion"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
    )
    assertion_type = sa.Column(sa.String)
    lower_id = sa.Column(sa.Integer, sa.ForeignKey("object.id"))
    greater_id = sa.Column(sa.Integer, sa.ForeignKey("object.id"))

    lower = sa.orm.relationship(
        "Object",
        foreign_keys=[lower_id],
        backref=sa.orm.backref(
            "greater_assertions",
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )
    greater = sa.orm.relationship(
        "Object",
        foreign_keys=[greater_id],
        backref=sa.orm.backref(
            "lower_assertions",
            cascade="all, delete-orphan",
            passive_deletes=True,
        ),
    )

    __mapper_args__ = {"polymorphic_identity": "assertion"}

    @classmethod
    def _from_object(cls, assertion):
        return Assertion(
            assertion_type=assertion.__class__.__name__,
            lower=Object.from_object(assertion.lower),
            greater=Object.from_object(assertion.greater),
        )
