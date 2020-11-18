from sqlalchemy import Column, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

_schema_version = 1


class PriorModel(Base):
    __tablename__ = "prior_model"

    id = Column(
        Integer,
        primary_key=True
    )
