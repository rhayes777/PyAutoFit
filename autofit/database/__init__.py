import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from autoconf import conf
from .aggregator import *
from .model import *


def open_database(
        filename="database.sqlite"
) -> Session:
    """
    Open a database file in the output directory

    Parameters
    ----------
    filename
        The name for the database file including sqlite suffix

    Returns
    -------
    A SQLAlchemy session
    """
    output_path = conf.instance.output_path

    os.makedirs(
        output_path,
        exist_ok=True
    )

    engine = create_engine(
        f'sqlite:///{output_path}/{filename}'
    )
    session = sessionmaker(bind=engine)()
    Base.metadata.create_all(engine)
    return session
