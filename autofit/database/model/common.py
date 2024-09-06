from typing import Union

from ..sqlalchemy_ import sa

from autofit.mapper.prior import abstract
from autofit.mapper.prior_model import prior_model
from autofit.mapper.prior_model import collection

from .model import Object


class Dict(Object):
    """
    Represents a Python dictionary in the database
    """

    __tablename__ = "dict"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )

    __mapper_args__ = {"polymorphic_identity": "dict"}

    def __call__(self):
        return {child.name: child() for child in self.children}

    @classmethod
    def _from_object(cls, source: dict):
        instance = cls()
        instance._add_children(source.items())
        instance.cls = dict
        return instance
