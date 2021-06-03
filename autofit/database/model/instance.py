from sqlalchemy import Column, Integer, String, ForeignKey, Float

from .model import Object


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
        'polymorphic_identity': 'none'
    }

    def __call__(self):
        return None


class Collection(Object):
    """
    A tuple or list
    """

    __tablename__ = "collection"

    id = Column(
        Integer,
        ForeignKey(
            "object.id"
        ),
        primary_key=True
    )

    __mapper_args__ = {
        'polymorphic_identity': 'collection'
    }

    @classmethod
    def _from_object(
            cls,
            source
    ):
        instance = cls()
        instance.cls = type(source)
        instance._add_children([
            (str(i), item)
            for i, item in enumerate(
                source
            )
        ])
        return instance

    def __call__(self):
        return self.cls([
            child()
            for child
            in sorted(
                self.children,
                key=lambda child: int(child.name)
            )
        ])


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
        if hasattr(
                source,
                "__getstate__"
        ):
            instance._add_children(
                source.__getstate__().items()
            )
        elif hasattr(
                source,
                "__dict__"
        ):
            instance._add_children(
                source.__dict__.items()
            )
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

    def __call__(self):
        return self.value
