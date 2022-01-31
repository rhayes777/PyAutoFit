import pickle
from functools import wraps
from typing import List

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import Samples
from .model import Base, Object


class Pickle(Base):
    """
    A pickled python object that was found in the pickles directory
    """

    try:

        from sqlalchemy.orm import relationship
        from sqlalchemy import Column, Integer, ForeignKey, String

        __tablename__ = "pickle"

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

    except ModuleNotFoundError:

        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        try:
            self.string = pickle.dumps(
                value
            )
        except pickle.PicklingError:
            pass


class Info(Base):

    try:

        from sqlalchemy.orm import relationship
        from sqlalchemy import Column, Integer, ForeignKey, String

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

    except ModuleNotFoundError:

        pass


def try_none(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError:
            return None

    return wrapper


class NamedInstance(Base):

    try:

        from sqlalchemy.orm import relationship
        from sqlalchemy import Column, Integer, ForeignKey, String

        __tablename__ = "named_instance"

        id = Column(
            Integer,
            primary_key=True
        )
        name = Column(String)

        instance_id = Column(
            Integer,
            ForeignKey(
                "object.id"
            )
        )

        __instance = relationship(
            "Object",
            uselist=False,
            backref="named_instance",
            foreign_keys=[instance_id]
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

    except ModuleNotFoundError:

        pass

    @property
    @try_none
    def instance(self):
        """
        An instance of the model labelled with a given name
        """
        return self.__instance()

    @instance.setter
    def instance(self, instance):
        self.__instance = Object.from_object(
            instance
        )


# noinspection PyProtectedMember
class NamedInstancesWrapper:
    def __init__(self, fit: "Fit"):
        """
        Provides dictionary like interface for accessing
        instance objects

        Parameters
        ----------
        fit
            A fit from which instances are accessed
        """
        self.fit = fit

    def __getitem__(self, item: str):
        """
        Get an instance with a given name.

        Raises a KeyError if no such instance exists.
        """
        return self._get_named_instance(
            item
        ).instance

    def __setitem__(self, key: str, value):
        """
        Set an instance for a given name
        """
        try:
            named_instance = self._get_named_instance(
                key
            )
        except KeyError:
            named_instance = NamedInstance(
                name=key
            )
            self.fit._named_instances.append(
                named_instance
            )
        named_instance.instance = value

    def _get_named_instance(
            self,
            item: str
    ) -> "NamedInstance":
        """
        Retrieve a NamedInstance by its name.
        """
        for named_instance in self.fit._named_instances:
            if named_instance.name == item:
                return named_instance
        raise KeyError(
            f"Instance {item} not found"
        )


class Fit(Base):

    try:

        from sqlalchemy.orm import relationship, backref
        from sqlalchemy import Column, Integer, ForeignKey, String, Boolean, Float

        __tablename__ = "fit"

        id = Column(
            String,
            primary_key=True,
        )
        is_complete = Column(
            Boolean
        )

        _named_instances: List[NamedInstance] = relationship(
            "NamedInstance"
        )

        max_log_likelihood = Column(
            Float
        )

        parent_id = Column(
            String,
            ForeignKey(
                "fit.id"
            )
        )

        children: List["Fit"] = relationship(
            "Fit",
            backref=backref(
                'parent',
                remote_side=[id]
            )
        )

        is_grid_search = Column(
            Boolean
        )

        unique_tag = Column(
            String
        )
        name = Column(
            String
        )
        path_prefix = Column(
            String
        )

        _samples = relationship(
            Object,
            uselist=False,
            foreign_keys=[
                Object.samples_for_id
            ]
        )

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

        _info: List[Info] = relationship(
            "Info"
        )

        pickles: List[Pickle] = relationship(
            "Pickle",
            lazy="joined"
        )

    except ModuleNotFoundError:

        pass

    @property
    @try_none
    def instance(self):
        """
        The instance of the model that had the highest likelihood
        """
        return self.__instance()

    @instance.setter
    def instance(self, instance):
        self.__instance = Object.from_object(
            instance
        )

    @property
    def named_instances(self):
        return NamedInstancesWrapper(
            self
        )

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )

    @property
    def best_fit(self) -> "Fit":
        """
        Only for grid searches. Returns the child search with
        the highest log likelihood.
        """
        if not self.is_grid_search:
            raise TypeError(
                f"Fit {self.id} is not a grid search"
            )
        if len(self.children) == 0:
            raise TypeError(
                f"Grid search fit {self.id} has no children"
            )

        best_fit = None
        max_log_likelihood = float("-inf")

        for fit in self.children:
            if fit.max_log_likelihood > max_log_likelihood:
                best_fit = fit
                max_log_likelihood = fit.max_log_likelihood

        return best_fit

    @property
    @try_none
    def samples(self) -> Samples:
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

    @model.setter
    def model(self, model: AbstractPriorModel):
        self.__model = Object.from_object(
            model
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

    def value(self, name: str):
        try:
            return self.__getitem__(item=name)
        except AttributeError:
            return None

    @classmethod
    def all(cls, session):
        return session.query(
            cls
        ).all()

    def __str__(self):
        return self.id

    def __repr__(self):
        return f"<{self.__class__.__name__} {self}>"


try:
    from sqlalchemy import inspect
    fit_attributes = inspect(Fit).columns
except ModuleNotFoundError:
    pass


