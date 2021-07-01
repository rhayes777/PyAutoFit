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
    Open a database file in the output directory or connect to an
    existing database.

    If filename ends with ".sqlite" then an existing database file
    is opened or a new one is created.

    Otherwise, filename is assumed to describe the URL for an existing
    database which is connected to.

    To connect to a postgres database one must be created.
    1) Install postgres https://www.postgresql.org/download/
    2) Create a user
        createuser autofit
    3) Create a database
        createdb -O autofit autofit
    4) Open that database using this function
        open_database(
            "postgresql://autofit@localhost/autofit"
        )

    Note that the above instructions create a database called autofit
    with a user called autofit. You can create a new database for the
    same user:

        createdb -O autofit new_database

    Also note that the user has no password. This is fine for developing
    locally but if you ever find yourself making a database that other
    people might be able to access you might want to give the user a
    password.

    Parameters
    ----------
    filename
        The name for the database file including sqlite suffix

    Returns
    -------
    A SQLAlchemy session
    """
    if filename.endswith(
            ".sqlite"
    ):
        output_path = conf.instance.output_path

        os.makedirs(
            output_path,
            exist_ok=True
        )
        string = f'sqlite:///{output_path}/{filename}'
        kwargs = dict(
            connect_args={'timeout': 15}
        )
    else:
        string = filename
        kwargs = dict()

    engine = create_engine(
        string,
        **kwargs
    )
    session = sessionmaker(bind=engine)()
    Base.metadata.create_all(engine)
    return session
