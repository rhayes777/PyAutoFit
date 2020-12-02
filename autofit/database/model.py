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
    def priors(self) -> List["Prior"]:
        """
        A list of prior database representations attached to this object
        """
        return [
            child
            for child
            in self.children
            if isinstance(child, Prior)
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
        """
        Create a database object for an object in a model.

        The specific database class used depends on the type of
        the object.

        Parameters
        ----------
        source
            A model
        kwargs
            Additional arguments specified in the creation of an object

        Returns
        -------
        An instance of a concrete child of this class
        """
        if source is None:
            return object.__new__(NoneInstance)
        if isinstance(source, af.PriorModel):
            return object.__new__(PriorModel)
        if isinstance(source, af.Prior):
            return object.__new__(Prior)
        if isinstance(source, (float, int)):
            return object.__new__(Value)
        if isinstance(source, (af.CollectionPriorModel, dict, list)):
            return object.__new__(CollectionPriorModel)
        if isinstance(source, str):
            return object.__new__(StringValue)
        return object.__new__(Instance)

    def _make_instance(self) -> object:
        """
        Create the real instance for this object
        """
        return self.cls()

    def __call__(self):
        """
        Create the real instance for this object, with child
        attributes attached
        """
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
        """
        Add database representations of child attributes

        Parameters
        ----------
        items
            Attributes such as floats or priors that are associated
            with the real object
        """
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
    def _class_path_array(self) -> List[str]:
        """
        A list of strings describing the module and class of the
        real object represented here
        """
        return self.class_path.split(".")

    @property
    def _class_name(self) -> str:
        """
        The name of the real class
        """
        return self._class_path_array[-1]

    @property
    def _module_path(self) -> str:
        """
        The path of the module containing the real class
        """
        return ".".join(self._class_path_array[:-1])

    @property
    def _module(self):
        """
        The module containing the real class
        """
        return importlib.import_module(
            self._module_path
        )

    @property
    def cls(self) -> type:
        """
        The class of the real object
        """
        return getattr(
            self._module,
            self._class_name
        )

    @cls.setter
    def cls(self, cls: type):
        self.class_path = re.search("'(.*)'", str(cls))[1]


class NoneInstance(Object):
    __tablename__ = "none"

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
            _,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )

    def _make_instance(self) -> None:
        return None


class Instance(Object):
    """
    An instance, such as a class instance
    """

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
    """
    A float
    """

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


class StringValue(Object):
    """
    A string
    """

    __tablename__ = "string_value"

    __mapper_args__ = {
        'polymorphic_identity': 'string_value'
    }

    id = Column(
        Integer,
        ForeignKey(
            "object.id"
        ),
        primary_key=True,
    )

    value = Column(String)

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

    def __init__(
            self,
            collection: Union[
                af.CollectionPriorModel,
                list,
                dict
            ],
            **kwargs
    ):
        super().__init__(**kwargs)
        if not isinstance(
                collection,
                af.CollectionPriorModel
        ):
            collection = af.CollectionPriorModel(
                collection
            )
        self._add_children(
            collection.items()
        )
        self.cls = af.CollectionPriorModel


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

    def __init__(
            self,
            model: af.PriorModel,
            **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self.cls = model.cls
        self._add_children(model.items())

    def _make_instance(self):
        return af.PriorModel(
            self.cls
        )


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
