import pickle
from typing import List

from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship

from autofit import AbstractPriorModel
from .model import Base, Object


class Pickle(Base):
    """
    A pickled python object that was found in the pickles directory
    """

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
        """
        The unpickled object
        """
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
    dataset_name = Column(
        String
    )
    phase_name = Column(
        String
    )

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

    @property
    def model(self) -> AbstractPriorModel:
        """
        The model that was fit
        """
        return self.__model()

    @property
    def instance(self):
        """
        The instance of the model that had the highest likelihood
        """
        return self.__instance()

    @model.setter
    def model(self, model: AbstractPriorModel):
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

    def __getitem__(self, item: str):
        """
        Retrieve an object that was a pickle

        Parameters
        ----------
        item
            The name of the pickle.

            e.g. if the file were 'samples.pickle' then 'samples' would
            retrieve the unpickled object.

        Returns
        -------
        An unpickled object
        """
        for p in self.pickles:
            if p.name == item:
                return p.value
        return getattr(
            self,
            item
        )

    def __setitem__(
            self,
            key: str,
            value
    ):
        """
        Add a pickle.

        If a deserialised object is given then it is serialised
        before being added to the database.

        Parameters
        ----------
        key
            The name of the pickle
        value
            A string, bytes or object
        """
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
