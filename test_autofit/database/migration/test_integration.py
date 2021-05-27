import pytest

from autofit.database.migration import SessionWrapper


@pytest.fixture(
    autouse=True
)
def create_table(
        session
):
    session.execute(
        "CREATE TABLE test (id INTEGER PRIMARY KEY)"
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
