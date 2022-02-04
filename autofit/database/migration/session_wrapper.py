import logging
from functools import wraps
from typing import Optional
from ..sqlalchemy_ import sa

logger = logging.getLogger(
    __name__
)


def needs_revision_table(
        func
):
    """
    Applies to functions that depend on the existence
    of the revision table. If the table does not exist
    it is created and then the function is executed.

    If the table already existed but an OperationalError
    is raised then that error is propagated.

    Parameters
    ----------
    func
        Some function that depends on the revision table

    Returns
    -------
    A decorated function
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except sa.exc.OperationalError as e:
            if self.is_table:
                raise e
            self._init_revision_table()
            return func(self, *args, **kwargs)

    return wrapper


class SessionWrapper:
    def __init__(self, session: sa.orm.Session):
        """
        Wraps a SQLAlchemy session so that certain commands can be
        encapsulated.

        Parameters
        ----------
        session
        """
        self.session = session

    def _init_revision_table(self):
        """
        Creates the revision table with a single null entry
        """
        self.session.execute(
            "CREATE TABLE revision (revision_id VARCHAR PRIMARY KEY)"
        )
        self.session.execute(
            "INSERT INTO revision (revision_id) VALUES (null)"
        )

    @property
    def is_table(self) -> bool:
        """
        Does the revision table exist?
        """
        try:
            self.session.execute(
                "SELECT 1 FROM revision"
            )
            return True
        except sa.exc.OperationalError:
            return False

    @property
    @needs_revision_table
    def revision_id(self) -> Optional[str]:
        """
        Describes the current revision of the database. None if no
        revisions have been made.
        """
        for row in self.session.execute(
                "SELECT revision_id FROM revision"
        ):
            return row[0]
        return None

    @revision_id.setter
    @needs_revision_table
    def revision_id(self, revision_id: str):
        self.session.execute(
            sa.text(
                f"UPDATE revision SET revision_id = :revision_id"
            ),
            {"revision_id": revision_id}
        )
