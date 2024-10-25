import os
import pickle
from pathlib import Path

import pytest

import autofit as af
from autoconf.conf import output_path_for_test
from autofit.non_linear.paths.null import NullPaths


def test_null_paths():
    search = af.DynestyStatic()

    assert isinstance(search.paths, NullPaths)


class TestPathDecorator:
    @staticmethod
    def assert_paths_as_expected(paths):
        assert paths.name == "name"
        assert paths.path_prefix == ""

    def test_with_arguments(self):
        search = af.m.MockSearch()
        search.paths = af.DirectoryPaths(name="name")

        self.assert_paths_as_expected(search.paths)

    def test_positional(self):
        search = af.m.MockSearch("name")
        paths = search.paths

        assert paths.name == "name"

    def test_paths_argument(self):
        search = af.m.MockSearch()
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)

    def test_combination_argument(self):
        search = af.m.MockSearch(
            "other",
        )
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)


output_path = Path(__file__).parent / "path"


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.Gaussian)


@output_path_for_test(output_path)
def test_identifier_file(model):
    paths = af.DirectoryPaths()
    paths.model = model
    paths.search = af.DynestyStatic()
    paths.save_all({}, {})

    assert os.path.exists(output_path / paths.identifier / ".identifier")


def test_serialize(model):
    paths = af.DirectoryPaths()
    paths.model = model

    pickled_paths = pickle.loads(pickle.dumps(paths))

    assert pickled_paths.model is not None


def test_unique_tag():
    paths = af.DirectoryPaths(unique_tag="unique_tag")
    assert "unique_tag" in paths.output_path.parts
