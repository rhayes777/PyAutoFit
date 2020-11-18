import importlib
import re
from typing import List, Tuple, Any, Iterable, Union, ItemsView

from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

import autofit as af

Base = declarative_base()

_schema_version = 1


class Object(Base):
    __tablename__ = "object"

    type = Column(
        String
    )

    id = Column(
        Integer,
        primary_key=True,
    )

    parent_id = Column(
        Integer,
        ForeignKey(
            "object.id"
        )
    )
    parent = relationship(
        "Object",
        uselist=False
    )
    children: List["Object"] = relationship(
        "Object"
    )

    name = Column(String)

    @property
    def priors(self):
        return [
            child
            for child
            in self.children
            if isinstance(
                child,
                Prior
            )
        ]

    __mapper_args__ = {
        'polymorphic_identity': 'object',
        'polymorphic_on': type
    }

    def __new__(
            cls,
            source,
            **kwargs
    ):
        if isinstance(source, af.PriorModel):
            return object.__new__(PriorModel)
        if isinstance(source, af.Prior):
            return object.__new__(Prior)
        if isinstance(source, (float, int)):
            return object.__new__(Value)
        if isinstance(source, af.CollectionPriorModel):
            return object.__new__(CollectionPriorModel)
        return object.__new__(Instance)

    def _make_instance(self):
        return self.cls()

    def __call__(self):
        instance = self._make_instance()
        for child in self.children:
            setattr(
                instance,
                child.name,
                child()
            )
        return instance

    def _add_children(
            self,
            items: Union[
                ItemsView[str, Any],
                Iterable[Tuple[str, Any]]
            ]
    ):
        for key, value in items:
            self.children.append(
                Object(
                    value,
                    name=key
                )
            )

    class_path = Column(
        String
    )

    @property
    def _class_path_array(self):
        return self.class_path.split(".")

    @property
    def _class_name(self):
        return self._class_path_array[-1]

    @property
    def _module_path(self):
        return ".".join(self._class_path_array[:-1])

    @property
    def _module(self):
        return importlib.import_module(
            self._module_path
        )

    @property
    def cls(self):
        return getattr(
            self._module,
            self._class_name
        )

    @cls.setter
    def cls(self, cls):
        self.class_path = re.search("'(.*)'", str(cls))[1]


class Instance(Object):
    __tablename__ = "instance"

    id = Column(
        Integer,
        ForeignKey(
            "object.id"
        ),
        primary_key=True,
    )

    __mapper_args__ = {
        'polymorphic_identity': 'instance'
    }

    def __init__(
            self,
            model,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.cls = type(model)
        self._add_children(model.__dict__.items())


class Value(Object):
    __tablename__ = "value"

    __mapper_args__ = {
        'polymorphic_identity': 'value'
    }

    id = Column(
        Integer,
        ForeignKey(
            "object.id"
        ),
        primary_key=True,
    )

    value = Column(Float)

    def __init__(
            self,
            value,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.value = value

    def __call__(self):
        return self.value


class CollectionPriorModel(Object):
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

    def __init__(
            self,
            collection: af.CollectionPriorModel,
            **kwargs
    ):
        super().__init__(**kwargs)
        self._add_children(collection.items())
        self.cls = af.CollectionPriorModel


class PriorModel(Object):
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

    def __init__(
            self,
            model: af.PriorModel,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.cls = model.cls
        self._add_children(model.direct_prior_tuples)

    def _make_instance(self):
        return af.PriorModel(
            self.cls
        )


class Prior(Object):
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

    def __init__(
            self,
            model: af.Prior,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.cls = type(model)
        self._add_children(model.__dict__.items())
