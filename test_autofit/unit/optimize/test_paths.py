import autofit as af


class TestPathDecorator:
    @staticmethod
    def assert_paths_as_expected(paths):
        assert paths.name == "phase_name"
        assert paths.tag == "phase_tag"
        assert paths.folders == ["phase_folders"]

    def test_with_arguments(self):
        optimizer = af.MockNLO(
            phase_name="phase_name",
            phase_tag="phase_tag",
            phase_folders=("phase_folders",),
        )
        self.assert_paths_as_expected(optimizer.paths)

    def test_positional(self):
        optimizer = af.MockNLO("phase_name")
        paths = optimizer.paths

        assert paths.name == "phase_name"

    def test_paths_argument(self):
        optimizer = af.MockNLO(
            paths=af.Paths(
                name="phase_name",
                tag="phase_tag",
                folders=("phase_folders",),
            )
        )
        self.assert_paths_as_expected(optimizer.paths)

    def test_combination_argument(self):
        optimizer = af.MockNLO(
            "other",
            paths=af.Paths(
                name="phase_name",
                tag="phase_tag",
                folders=("phase_folders",),
            ),
        )
        self.assert_paths_as_expected(optimizer.paths)
