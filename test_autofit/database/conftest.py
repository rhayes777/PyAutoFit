import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autofit import database as db


@pytest.fixture(name="session")
def make_session():
    engine = create_engine('sqlite://')
    session = sessionmaker(bind=engine)()
    db.Base.metadata.create_all(engine)
    yield session
    session.close()
    engine.dispose()
