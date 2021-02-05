from autofit.database import query_model as q
from .model import Object


class Aggregator:
    def __init__(self, session):
        self.session = session

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
