from abc import ABC, abstractmethod
from hashlib import md5

from sqlalchemy.exc import OperationalError


class Identifiable(ABC):
    @property
    @abstractmethod
    def id(self):
        pass

    def __eq__(self, other):
        if isinstance(
                other,
                Identifiable
        ):
            return self.id == other.id
        if isinstance(
                other,
                str
        ):
            return self.id == other
        return False


class Migrator:
    def __init__(self, *steps):
        self._steps = steps

    @property
    def revisions(self):
        for i in range(1, len(self._steps) + 1):
            yield Revision(
                self._steps[:i]
            )

    def get_steps(self, revision_id=None):
        for revision in self.revisions:
            if revision_id == revision.id:
                return (self.latest_revision - revision).steps

        return self._steps

    @property
    def latest_revision(self):
        return Revision(
            self._steps
        )

    def migrate(self, session):
        wrapper = SessionWrapper(
            session
        )
        steps = self.get_steps(
            wrapper.revision_id
        )


class SessionWrapper:
    def __init__(self, session):
        self.session = session

    def _init_revision_table(self):
        self.session.execute(
            "CREATE TABLE revision (revision_id VARCHAR PRIMARY KEY)"
        )
        self.session.execute(
            "INSERT INTO revision (revision_id) VALUES ('origin')"
        )

    @property
    def revision_id(self):
        try:
            return self.session.execute(
                "SELECT revision_id FROM revision"
            )
        except OperationalError:
            self._init_revision_table()
            return None

    @revision_id.setter
    def revision_id(self, revision_id):
        self.session.execute(
            f"UPDATE revision SET revision_id = {revision_id}"
        )


class Revision(Identifiable):
    def __init__(self, steps):
        self.steps = steps

    @property
    def id(self):
        return md5(
            ":".join(
                step.id for step
                in self.steps
            ).encode("utf-8")
        ).hexdigest()

    def __sub__(self, other):
        return Revision(tuple(
            step for step in self.steps
            if step not in other.steps
        ))


class Step(Identifiable):
    def __init__(self, string):
        self.string = string

    @property
    def id(self):
        return md5(
            self.string.encode(
                "utf-8"
            )
        ).hexdigest()
