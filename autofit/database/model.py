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

    # noinspection PyProtectedMember
    @classmethod
    def from_object(
            cls,
            source,
            name=None
    ):
        """
        Create a database object for an object in a model.

        The specific database class used depends on the type of
        the object.

        Parameters
        ----------
        source
            A model
        name
            The name of the object wrt its parent

        Returns
        -------
        An instance of a concrete child of this class
        """
        if source is None:
            instance = NoneInstance()
        elif isinstance(source, af.PriorModel):
            instance = PriorModel._from_object(
                source
            )
        elif isinstance(source, af.Prior):
            instance = Prior._from_object(
                source
            )
        elif isinstance(source, (float, int)):
            instance = Value._from_object(
                source
            )
        elif isinstance(source, (af.CollectionPriorModel, dict, list)):
            instance = CollectionPriorModel._from_object(
                source
            )
        elif isinstance(source, str):
            instance = StringValue._from_object(
                source
            )
        else:
            instance = Instance._from_object(
                source
            )
        instance.name = name
        return instance

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
                Object.from_object(
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

    @classmethod
    def _from_object(
            cls,
            source
    ):
        instance = cls()
        instance.cls = type(source)
        instance._add_children(source.__dict__.items())
        return instance


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

    @classmethod
    def _from_object(
            cls,
            source
    ):
        instance = cls()
        instance.value = source
        return instance

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

    @classmethod
    def _from_object(
            cls,
            source
    ):
        instance = cls()
        instance.value = source
        return instance


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

    @classmethod
    def _from_object(
            cls,
            source: Union[
                af.CollectionPriorModel,
                list,
                dict
            ]
    ):
        instance = cls()
        if not isinstance(
                source,
                af.CollectionPriorModel
        ):
            source = af.CollectionPriorModel(
                source
            )
        instance._add_children(
            source.items()
        )
        instance.cls = af.CollectionPriorModel
        return instance


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

    @classmethod
    def _from_object(
            cls,
            model: af.PriorModel,
    ):
        instance = cls()
        instance.cls = model.cls
        instance._add_children(model.items())
        return instance

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

    @classmethod
    def _from_object(
            cls,
            model: af.Prior
    ):
        instance = cls()
        instance.cls = type(model)
        instance._add_children(model.__dict__.items())
        return instance
