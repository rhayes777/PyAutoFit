from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

from .model import Base


class Fit(Base):
    __tablename__ = "fit"

    id = Column(
        Integer,
        primary_key=True,
    )

    model_id = Column(
        Integer,
        ForeignKey(
            "object.id"
        )
    )
    model = relationship(
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
    instance = relationship(
        "Object",
        uselist=False,
        backref="fit_instance",
        foreign_keys=[instance_id]
    )
