import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import autofit as af
from autofit import database as db

directory = Path(__file__).parent
database_path = f"{directory}/test.db"


@pytest.fixture(
    name="session",
    scope="module"
)
def make_session():
    engine = create_engine(f"sqlite:///{database_path}")
    session = sessionmaker(bind=engine)()
    db.Base.metadata.create_all(engine)
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(
    autouse=True,
    scope="module"
)
def read_in(session):
    aggregator = af.Aggregator(
        directory.parent.parent.parent.parent / "rjlens"
    )
    for item in aggregator:
        obj = db.Object.from_object(
            item.model
        )
        session.add(
            obj
        )
    yield
    try:
        os.remove(database_path)
    except FileNotFoundError:
        pass


def test_commit(session):
    session.commit()


def test_instantiate(session):
    model = session.query(
        db.Object
    ).filter(
        db.Object.parent_id.is_(None)
    ).first()
    assert isinstance(
        model(),
        af.AbstractPriorModel
    )
