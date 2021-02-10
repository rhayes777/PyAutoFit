import pickle
from typing import List

from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship

from .model import Base, Object


class Pickle(Base):
    __tablename__ = "pickle"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    id = Column(
        Integer,
        primary_key=True
    )

    name = Column(
        String
    )
    string = Column(
        String
    )
    fit_id = Column(
        Integer,
        ForeignKey(
            "fit.id"
        )
    )
    fit = relationship(
        "Fit",
        uselist=False
    )

    @property
    def value(self):
        return pickle.loads(
            self.string
        )

    @value.setter
    def value(self, value):
        self.string = pickle.dumps(
            value
        )


class Fit(Base):
    __tablename__ = "fit"

    id = Column(
        Integer,
        primary_key=True,
    )

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

    @property
    def model(self):
        return self.__model()

    @property
    def instance(self):
        return self.__instance()

    @model.setter
    def model(self, model):
        self.__model = Object.from_object(
            model
        )

    @instance.setter
    def instance(self, instance):
        self.__instance = Object.from_object(
            instance
        )

    pickles: List[Pickle] = relationship(
        "Pickle"
    )

    def __getitem__(self, item):
        for p in self.pickles:
            if p.name == item:
                return p.value
        return getattr(
            self,
            item
        )

    def __setitem__(self, key, value):
        new = Pickle(
            name=key
        )
        if isinstance(
                value,
                (str, bytes)
        ):
            new.string = value
        else:
            new.value = value
        self.pickles = [
                           p
                           for p
                           in self.pickles
                           if p.name != key
                       ] + [
                           new
                       ]

    model_id = Column(
        Integer,
        ForeignKey(
            "object.id"
        )
    )
    __model = relationship(
        "Object",
        uselist=False,
        backref="fit_model",
        foreign_keys=[model_id]
    )

    instance_id = Column(
        Integer,
        ForeignKey(
            "object.id"
        )
    )
    __instance = relationship(
        "Object",
        uselist=False,
        backref="fit_instance",
        foreign_keys=[instance_id]
    )
