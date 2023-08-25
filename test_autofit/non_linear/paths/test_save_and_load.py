import pytest

import autofit as af


@pytest.fixture(name="dictionary")
def make_dictionary():
    return {"hello": "world"}


def test_directory_json(dictionary):
    paths = af.DirectoryPaths()

    paths.save_json("test", dictionary)
    assert paths.load_json("test") == dictionary


def test_database_json(dictionary, session):
    paths = af.DatabasePaths(session)

    paths.save_json("test", dictionary)
    assert paths.load_json("test") == dictionary
