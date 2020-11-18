import importlib
import re

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

import autofit as af

Base = declarative_base()

_schema_version = 1


class PriorModel(Base):
    __tablename__ = "prior_model"

    id = Column(
        Integer,
        primary_key=True
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

    def __init__(self, model: af.PriorModel):
        self.class_path = re.search("'(.*)'", str(model.cls))[1]

    def __call__(self):
        return self.cls()
