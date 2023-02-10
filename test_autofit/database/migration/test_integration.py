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
        session,
        revision_2
):
    migrator.migrate(
        session
    )
    assert len(list(
        session.execute(
            "SELECT * FROM test"
        )
    )) == 2

    assert SessionWrapper(
        session
    ).revision_id == revision_2


def test_apply_twice(
        migrator,
        session
):
    for _ in range(2):
        migrator.migrate(
            session
        )
    assert len(list(
        session.execute(
            "SELECT * FROM test"
        )
    )) == 2


def test_run_partial_migration(
        migrator,
        session,
        revision_1,
        revision_2
):
    wrapper = SessionWrapper(
        session
    )
    wrapper.revision_id = revision_1.id

    assert len(migrator.get_steps(revision_1.id)) == 1

    migrator.migrate(
        session
    )
    assert len(list(
        session.execute(
            "SELECT * FROM test"
        )
    )) == 1

    assert wrapper.revision_id == revision_2


def test_session_wrapper(session):
    wrapper = SessionWrapper(session)
    assert wrapper.revision_id is None

    wrapper.revision_id = "revision_id"
    assert wrapper.revision_id == "revision_id"


def test_table_exists(session):
    wrapper = SessionWrapper(session)
    assert wrapper.is_table is False

    _ = wrapper.revision_id
    assert wrapper.is_table is True
