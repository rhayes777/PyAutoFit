import pickle
from functools import wraps
from typing import List

from sqlalchemy import Column, Integer, ForeignKey, String, Boolean, inspect
from sqlalchemy.orm import relationship

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import OptimizerSamples
from .model import Base, Object
from ...mapper.model_object import Identifier


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
        String,
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
        if isinstance(
                self.string,
                str
        ):
            return self.string
        return pickle.loads(
            self.string
        )

    @value.setter
    def value(self, value):
        self.string = pickle.dumps(
            value
        )


class Info(Base):
    __tablename__ = "info"

    id = Column(
        Integer,
        primary_key=True
    )

    key = Column(String)
    value = Column(String)

    fit_id = Column(
        String,
        ForeignKey(
            "fit.id"
        )
    )
    fit = relationship(
        "Fit",
        uselist=False
    )


def try_none(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            return None

    return wrapper


class Fit(Base):
    __tablename__ = "fit"

    id = Column(
        String,
        primary_key=True,
    )
    is_complete = Column(
        Boolean
    )

    _info: List[Info] = relationship(
        "Info"
    )

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )

    parent_id = Column(
        String,
        ForeignKey(
            "fit.id"
        )
    )
    parent: "Fit" = relationship(
        "Fit",
        uselist=False,
        foreign_keys=[
            parent_id
        ]
    )
    children = relationship(
        "Fit"
    )

    is_grid_search = Column(
        Boolean
    )

    unique_tag = Column(
        String
    )

    _samples = relationship(
        Object,
        uselist=False,
        foreign_keys=[
            Object.samples_for_id
        ]
    )

    @property
    def samples(self) -> OptimizerSamples:
        return self._samples()

    @samples.setter
    def samples(self, samples):
        self._samples = Object.from_object(
            samples
        )

    @property
    def info(self):
        return {
            info.key: info.value
            for info
            in self._info
        }

    @info.setter
    def info(self, info):
        if info is not None:
            self._info = [
                Info(
                    key=key,
                    value=value
                )
                for key, value
                in info.items()
            ]

    @property
    @try_none
    def model(self) -> AbstractPriorModel:
        """
        The model that was fit
        """
        return self.__model()

    @property
    @try_none
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

    def __contains__(self, item):
        for p in self.pickles:
            if p.name == item:
                return True
        return False

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

    def __delitem__(self, key):
        self.pickles = [
            p
            for p
            in self.pickles
            if p.name != key
        ]

    def value(self, name : str):
        try:
            return self.__getitem__(item=name)
        except AttributeError:
            return None

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

    @classmethod
    def all(cls, session):
        return session.query(
            cls
        ).all()

    def __str__(self):
        return self.id

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"


fit_attributes = inspect(Fit).columns
