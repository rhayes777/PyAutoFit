import os
import shutil
from pathlib import Path

import pytest

import autofit as af

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


def test():
    af.Aggregator.from_database(
        copy
    )
