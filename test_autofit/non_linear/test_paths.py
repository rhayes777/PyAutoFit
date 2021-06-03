import os

import autofit as af
from pathlib import Path

from autoconf.conf import output_path_for_test


class TestPathDecorator:
    @staticmethod
    def assert_paths_as_expected(paths):
        assert paths.name == "name"
        assert paths.path_prefix == ""

    def test_with_arguments(self):
        search = af.MockSearch()
        search.paths = af.DirectoryPaths(name="name")

        self.assert_paths_as_expected(search.paths)

    def test_positional(self):
        search = af.MockSearch("name")
        paths = search.paths

        assert paths.name == "name"

    def test_paths_argument(self):
        search = af.MockSearch()
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)

    def test_combination_argument(self):
        search = af.MockSearch("other", )
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)


output_path = Path(
    __file__
).parent / "path"


@output_path_for_test(
    output_path
)
def test_identifier_file():
    paths = af.DirectoryPaths()
    paths.model = af.Model(
        af.Gaussian
    )
    paths.search = af.DynestyStatic()

    assert os.path.exists(
        output_path / paths.identifier / ".identifier"
    )
