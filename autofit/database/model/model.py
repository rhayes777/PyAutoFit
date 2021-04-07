import inspect
from typing import List, Tuple, Any, Iterable, Union, ItemsView, Type

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

import autofit as af
from autofit.util import get_class_path, get_class

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
        remote_side=[id]
    )
    children: List["Object"] = relationship(
        "Object",
        uselist=True,
    )

    name = Column(String)

    __mapper_args__ = {
        'polymorphic_identity': 'object',
        'polymorphic_on': type
    }

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    # noinspection PyProtectedMember
    @classmethod
    def from_object(
            cls,
            source,
            name=""
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
            from .instance import NoneInstance
            instance = NoneInstance()
        elif isinstance(source, af.PriorModel):
            from .prior import PriorModel
            instance = PriorModel._from_object(
                source
            )
        elif isinstance(source, af.Prior):
            from .prior import Prior
            instance = Prior._from_object(
                source
            )
        elif isinstance(source, (float, int)):
            from .instance import Value
            instance = Value._from_object(
                source
            )
        elif isinstance(source, (tuple, list)):
            from .instance import Collection
            instance = Collection._from_object(
                source
            )
        elif isinstance(source, (af.CollectionPriorModel, dict, list)):
            from .prior import CollectionPriorModel
            instance = CollectionPriorModel._from_object(
                source
            )
        elif isinstance(source, str):
            from .instance import StringValue
            instance = StringValue._from_object(
                source
            )
        else:
            from .instance import Instance
            instance = Instance._from_object(
                source
            )
        instance.name = name
        return instance

    @property
    def _constructor_args(self):
        return set(
            inspect.getfullargspec(
                self.cls
            ).args[1:]
        )

    def _make_instance(self) -> object:
        """
        Create the real instance for this object
        """
        return object.__new__(self.cls)

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
            if isinstance(
                    value,
                    property
            ):
                continue
            child = Object.from_object(
                value,
                name=key
            )
            self.children.append(
                child
            )

    class_path = Column(
        String
    )

    @property
    def cls(self) -> Type[object]:
        """
        The class of the real object
        """
        return get_class(
            self.class_path
        )

    @cls.setter
    def cls(self, cls: type):
        self.class_path = get_class_path(cls)
