import logging
from functools import wraps

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(
    __name__
)


def needs_revision_table(
        func
):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except OperationalError:
            self._init_revision_table()
            return func(self, *args, **kwargs)

    return wrapper


class SessionWrapper:
    def __init__(self, session):
        self.session = session

    def _init_revision_table(self):
        self.session.execute(
            "CREATE TABLE revision (revision_id VARCHAR PRIMARY KEY)"
        )
        self.session.execute(
            "INSERT INTO revision (revision_id) VALUES (null)"
        )

    @property
    @needs_revision_table
    def revision_id(self):
        for row in self.session.execute(
                "SELECT revision_id FROM revision"
        ):
            return row[0]
        return None

    @revision_id.setter
    @needs_revision_table
    def revision_id(self, revision_id):
        self.session.execute(
            text(
                f"UPDATE revision SET revision_id = :revision_id"
            ),
            {"revision_id": revision_id}
        )
