from typing import Union, ItemsView, Any, Iterable, Tuple

from ..sqlalchemy_ import sa

from .model import Object


class Dict(Object):
    """
    Represents a Python dictionary in the database
    """

    __tablename__ = "dict"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )

    __mapper_args__ = {"polymorphic_identity": "dict"}

    def __call__(self):
        d = {}
        for child in self.children:
            instance = child()
            if child.name != "":
                d[child.name] = instance
            else:
                d[instance[0]] = instance[1]

        return d

    @classmethod
    def _from_object(cls, source: dict):
        instance = cls()
        instance._add_children(source.items())
        instance.cls = dict
        return instance

    def _add_children(
        self, items: Union[ItemsView[str, Any], Iterable[Tuple[str, Any]]]
    ):
        """
        Add database representations of child attributes

        Parameters
        ----------
        items
            Attributes such as floats or priors that are associated
            with the real object
        """
        self.children = [
            Object.from_object(value, name=key)
            if isinstance(key, str)
            else Object.from_object((key, value))
            for key, value in items
        ]
