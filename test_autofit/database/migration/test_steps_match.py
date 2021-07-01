import os
import shutil
from pathlib import Path

import pytest
from sqlalchemy import String, Column
from sqlalchemy.exc import OperationalError

import autofit as af
from autoconf.conf import output_path_for_test
from autofit.database import Fit

directory = Path(
    __file__
).parent


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
                "database_copy.sqlite"
            ).fits
        )
    except OperationalError as e:
        raise AssertionError(
            "Migration steps are not complete"
        ) from e


@output_path_for_test(
    str(directory),
    remove=False
)
def test():
    """
    Raises an exception if changes to autofit/database/migration/steps.py
    are required to migrate old database files correctly.
    """
    _check_migration()


@output_path_for_test(
    str(directory),
    remove=False
)
def test_fail():
    Fit.random_column = Column(
        String
    )

    with pytest.raises(
            AssertionError
    ):
        _check_migration()
