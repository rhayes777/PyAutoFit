from sqlalchemy import and_

from .instance import *
from .model import *
from .prior import *


class Aggregator:
    def __init__(self, session):
        self.session = session

    def filter(self, **kwargs):
        name, value = list(kwargs.items())[0]
        return self.session.query(
            Value
        ).filter(
            and_(
                Value.name == name,
                Value.value == value
            )
        ).one().parent
