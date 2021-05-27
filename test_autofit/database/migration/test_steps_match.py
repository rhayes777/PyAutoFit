import os
import shutil
from pathlib import Path

import pytest
from sqlalchemy import String, Column
from sqlalchemy.exc import OperationalError

import autofit as af
from autofit.database import Fit

origin = Path(
    __file__
).parent / "database.sqlite"

copy = Path(
    __file__
).parent / "database_copy.sqlite"


@pytest.fixture(
    autouse=True
)
def manage_files():
    shutil.copy(
        origin,
        copy
    )
    yield
    os.remove(
        copy,

    )


def _check_migration():
    try:
        print(
            af.Aggregator.from_database(
                copy
            ).fits
        )
    except OperationalError as e:
        raise AssertionError(
            "Migration steps are not complete"
        ) from e


def test():
    """
    Raises an exception if changes to autofit/database/migration/steps.py
    are required to migrate old database files correctly.
    """
    _check_migration()


def test_fail():
    Fit.random_column = Column(
        String
    )

    with pytest.raises(
            AssertionError
    ):
        _check_migration()
