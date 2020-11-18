import importlib
import re
from typing import List

from sqlalchemy import Column, Integer, String, ForeignKey
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
        uselist=False,
        # foreign_keys=[parent_id]
    )
    children: List["Object"] = relationship(
        "Object"
    )

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

    def __new__(cls, source):
        if isinstance(source, af.PriorModel):
            return object.__new__(PriorModel)
        if isinstance(source, af.Prior):
            return object.__new__(Prior)
        raise TypeError(
            f"{type(source)} is not supported"
        )

    def __call__(self):
        raise NotImplementedError()


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

    def __init__(self, model: af.PriorModel):
        self.class_path = re.search("'(.*)'", str(model.cls))[1]
        for name, prior in model.direct_prior_tuples:
            self.children.append(
                Object(
                    prior
                )
            )

    def __call__(self):
        return self.cls()


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

    def __init__(self, model: af.Prior):
        pass

    def __call__(self):
        pass
