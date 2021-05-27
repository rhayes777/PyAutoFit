import pytest
from sqlalchemy.exc import OperationalError

from autofit.database.migration import Migrator, SessionWrapper


def test_session_wrapper(session):
    wrapper = SessionWrapper(session)
    assert wrapper.revision_id is None

    wrapper.revision_id = "revision_id"
    assert wrapper.revision_id == "revision_id"


def test_creates_table(session):
    migrator = Migrator()

    with pytest.raises(
            OperationalError
    ):
        session.execute(
            "SELECT revision_id FROM revision"
        )

    migrator.migrate(session)

    assert len(list(
        session.execute(
            "SELECT revision_id FROM revision"
        )
    )) == 1
