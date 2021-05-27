import pytest
from sqlalchemy.exc import OperationalError

from autofit.database.migration import Migrator, SessionWrapper, Step


@pytest.fixture(
    name="migrator"
)
def make_migrator():
    return Migrator(
        Step(
            "CREATE TABLE test (id INTEGER PRIMARY KEY)"
        ),
        Step(
            "INSERT INTO test (id) VALUES (1)"
        ),
        Step(
            "INSERT INTO test (id) VALUES (2)"
        )
    )


def test_run_migration(
        migrator,
        session
):
    migrator.migrate(
        session
    )
    assert len(list(
        session.execute(
            "SELECT * FROM test"
        )
    )) == 2


def test_session_wrapper(session):
    wrapper = SessionWrapper(session)
    assert wrapper.revision_id is None

    wrapper.revision_id = "revision_id"
    assert wrapper.revision_id == "revision_id"


def test_creates_table(session, migrator):
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
