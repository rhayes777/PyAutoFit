from typing import Union

from sqlalchemy import Column, Integer, ForeignKey

import autofit as af
from .model import Object


class CollectionPriorModel(Object):
    """
    A collection
    """

    __tablename__ = "collection_prior_model"

    id = Column(
        Integer,
        ForeignKey(
            "object.id"
        ),
        primary_key=True,
    )

    __mapper_args__ = {
        'polymorphic_identity': 'collection_prior_model'
    }

    @classmethod
    def _from_object(
            cls,
            source: Union[
                af.CollectionPriorModel,
                list,
                dict
            ]
    ):
        instance = cls()
        if not isinstance(
                source,
                af.CollectionPriorModel
        ):
            source = af.CollectionPriorModel(
                source
            )
        instance._add_children(
            source.items()
        )
        instance.cls = af.CollectionPriorModel
        return instance


class PriorModel(Object):
    """
    A prior model
    """

    __tablename__ = "prior_model"

    id = Column(
        Integer,
        ForeignKey(
            "object.id"
        ),
        primary_key=True,
    )

    __mapper_args__ = {
        'polymorphic_identity': 'prior_model'
    }

    @classmethod
    def _from_object(
            cls,
            model: af.PriorModel,
    ):
        instance = cls()
        instance.cls = model.cls
        instance._add_children(model.items())
        return instance

    def _make_instance(self):
        instance = object.__new__(af.PriorModel)
        instance.cls = self.cls
        return instance


class Prior(Object):
    """
    A prior
    """

    __tablename__ = "prior"

    id = Column(
        Integer,
        ForeignKey(
            "object.id"
        ),
        primary_key=True,
    )

    __mapper_args__ = {
        'polymorphic_identity': 'prior'
    }

    @classmethod
    def _from_object(
            cls,
            model: af.Prior
    ):
        instance = cls()
        instance.cls = type(model)
        instance._add_children(model.__dict__.items())
        return instance
