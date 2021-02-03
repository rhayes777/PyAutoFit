from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autofit.aggregator.aggregator import Aggregator as ClassicAggregator
from autofit.database import query_model as q
from .model import Object, Base


class Aggregator:
    def __init__(
            self,
            session,
            filename=None
    ):
        self.session = session
        self.filename = filename

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.filename}>"

    def __getattr__(self, name):
        return q.Q(name)

    def filter(self, predicate):
        objects_ids = {
            row[0]
            for row
            in self.session.execute(
                predicate.query
            )
        }
        return self.session.query(
            Object
        ).filter(
            Object.id.in_(
                objects_ids
            )
        ).all()

    def add_directory(self, directory):
        aggregator = ClassicAggregator(
            directory
        )
        for item in aggregator:
            obj = Object.from_object(
                item.model
            )
            self.session.add(
                obj
            )
        self.session.commit()

    @classmethod
    def from_database(cls, filename):
        engine = create_engine(
            f'sqlite:///{filename}'
        )
        session = sessionmaker(
            bind=engine
        )()
        Base.metadata.create_all(
            engine
        )
        return Aggregator(
            session,
            filename
        )
