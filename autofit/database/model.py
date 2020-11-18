import importlib
import re

from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

import autofit as af

Base = declarative_base()

_schema_version = 1


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


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
    def class_path_array(self):
        return self.class_path.split(".")

    @property
    def class_name(self):
        return self.class_path_array[-1]

    @property
    def module_path(self):
        return ".".join(self.class_path_array[:-1])

    @property
    def module(self):
        return importlib.import_module(
            self.module_path
        )

    @property
    def cls(self):
        return getattr(
            self.module,
            self.class_name
        )

    def __init__(self, model: af.PriorModel):
        self.class_path = re.search("'(.*)'", str(model.cls))[1]

    def __call__(self):
        return self.cls()
