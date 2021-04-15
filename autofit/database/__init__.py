from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .aggregator import *
from .model import *


def open_database(
        filename
):
    engine = create_engine(
        f'sqlite:///{filename}'
    )
    session = sessionmaker(bind=engine)()
    Base.metadata.create_all(engine)
    return session
