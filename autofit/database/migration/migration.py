import logging
from abc import ABC, abstractmethod
from hashlib import md5
from sqlalchemy import text
from typing import Union, Generator, Iterable, Optional

from .session_wrapper import SessionWrapper
from ..sqlalchemy_ import sa

logger = logging.getLogger(__name__)


class Identifiable(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """
        A unique identifier generated by hashing a string
        """

    def __eq__(self, other: Union["Identifiable", str]) -> bool:
        """
        Compares ids
        """
        if isinstance(other, Identifiable):
            return self.id == other.id
        if isinstance(other, str):
            return self.id == other
        return False


class Step(Identifiable):
    def __init__(self, *strings: str):
        """
        A collection of SQL statements executed as one step
        in a database migration.

        Parameters
        ----------
        strings
            SQL statements
        """
        self.strings = strings

    @property
    def id(self) -> str:
        """
        Hash generated from underlying SQL statements
        """
        return md5(":".join(self.strings).encode("utf-8")).hexdigest()

    def __str__(self):
        return "\n".join(self.strings)

    __repr__ = __str__


class Revision(Identifiable):
    def __init__(self, steps: Iterable[Step]):
        """
        A specific revision of the database. This comprises
        a set of sequential steps and is uniquely identified
        by a hash on the hash of those steps.

        Parameters
        ----------
        steps
            Collections of SQL statements describing the changes
            between different versions of the database
        """
        self.steps = steps

    @property
    def id(self) -> str:
        """
        A unique identifier created by joining and hashing the
        identifiers of comprised steps.
        """
        return md5(":".join(step.id for step in self.steps).encode("utf-8")).hexdigest()

    def __sub__(self, other: "Revision") -> "Revision":
        """
        Create a revision with steps that describe the difference
        between two revisions.

        For example, if the data base were at revision 2 and the
        code at revision 5, a 'revision' would be returned containing
        the steps required to migrate the database to revision 5.

        Parameters
        ----------
        other
            A previous revision

        Returns
        -------
        An object comprising steps required to move from the other
        revision to this revision.
        """
        return Revision(tuple(step for step in self.steps if step not in other.steps))


class Migrator:
    def __init__(self, *steps: Step):
        """
        Manages migration of an old database.

        The revision table is checked to see what version a database is on.
        This is compared to the identifier of the current revision to determine
        the set of Steps that must be executed to migrate the database.

        Parameters
        ----------
        steps
            All steps recorded for every migration
        """
        self._steps = steps

    @property
    def revisions(self) -> Generator[Revision, None, None]:
        """
        One revision exists for each sequential set of steps
        starting on the first step and terminating on any step
        """
        for i in range(1, len(self._steps) + 1):
            yield Revision(self._steps[:i])

    def get_steps(self, revision_id: Optional[str] = None) -> Iterable[Step]:
        """
        Retrieve steps required to go from the specified
        revision to the latest revision.

        Parameters
        ----------
        revision_id
            The identifier for a revision.

            If None or unrecognised then all steps are returned.

        Returns
        -------
        Steps required to get to the latest revision.
        """
        for revision in self.revisions:
            if revision_id == revision.id:
                return (self.latest_revision - revision).steps

        return self._steps

    @property
    def latest_revision(self) -> Revision:
        """
        The latest revision according to the steps passed to the
        Migrator
        """
        return Revision(self._steps)

    def migrate(self, session: sa.orm.Session):
        """
        Migrate the database that session points to to the current
        revision.

        Applies each required step and updates the revision identifier
        in the database.

        If no revision table is found then one is created.

        Parameters
        ----------
        session
            A session pointing at some database.
        """
        wrapper = SessionWrapper(session)
        revision_id = wrapper.revision_id
        steps = list(self.get_steps(revision_id))
        if len(steps) == 0:
            logger.info("Database already at latest revision")
            return

        latest_revision_id = self.latest_revision.id

        logger.info(
            f"Performing migration from {revision_id} to {latest_revision_id} in {len(steps)} steps"
        )
        for step in steps:
            for string in step.strings:
                try:
                    session.execute(text(string))
                except sa.exc.OperationalError as e:
                    logger.debug(e)

        wrapper.revision_id = self.latest_revision.id

        logger.info(f"revision_id updated to {wrapper.revision_id}")
