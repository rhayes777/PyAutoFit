import pytest
from sqlalchemy.exc import OperationalError

from autofit.database.migration import Migrator


# @pytest.fixture(
#     name="session",
#     scope="module"
# )
# def make_session():
#     engine = create_engine('sqlite://')
#     session = sessionmaker(bind=engine)()
#     db.Base.metadata.create_all(engine)
#     yield session
#     session.close()
#     engine.dispose()


def test_creates_table(session):
    migrator = Migrator()

    with pytest.raises(
            OperationalError
    ):
        session.execute(
            "SELECT revision_id FROM revision"
        )

    migrator.migrate(session)

    result = session.execute(
        "SELECT revision_id FROM revision"
    )
    print(result)

    # assert False
# self.session.execute
