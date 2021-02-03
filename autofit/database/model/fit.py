from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

from .model import Base


class Fit(Base):
    __tablename__ = "fit"

    id = Column(
        Integer,
        primary_key=True,
    )

    model = relationship(
        "Object",
        uselist=False,
        backref="fit"
    )
    model_id = Column(
        Integer,
        ForeignKey(
            "object.id"
        )
    )
