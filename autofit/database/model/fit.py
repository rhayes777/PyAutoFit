import pickle

from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship

from .model import Base, Object


class Fit(Base):
    __tablename__ = "fit"

    id = Column(
        Integer,
        primary_key=True,
    )

    @property
    def model(self):
        return self.__model()

    @property
    def instance(self):
        return self.__instance()

    @property
    def samples(self):
        return pickle.loads(self.__samples)

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

    @samples.setter
    def samples(self, samples):
        self.__samples = pickle.dumps(samples)

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

    __samples = Column(String)
