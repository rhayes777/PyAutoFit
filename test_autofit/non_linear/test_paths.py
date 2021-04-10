import autofit as af


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
