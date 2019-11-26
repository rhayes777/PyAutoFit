import autofit as af


class TestPathDecorator:
    @staticmethod
    def assert_paths_as_expected(paths):
        assert paths.phase_name == "phase_name"
        assert paths.phase_tag == "phase_tag"
        assert paths.phase_folders == ["phase_folders"]

    def test_with_arguments(self):
        optimizer = af.NonLinearOptimizer(
            phase_name="phase_name",
            phase_tag="phase_tag",
            phase_folders=("phase_folders",),
        )
        self.assert_paths_as_expected(optimizer.paths)

    def test_positional(self):
        optimizer = af.NonLinearOptimizer("phase_name")
        paths = optimizer.paths

        assert paths.phase_name == "phase_name"

    def test_paths_argument(self):
        optimizer = af.NonLinearOptimizer(
            paths=af.Paths(
                phase_name="phase_name",
                phase_tag="phase_tag",
                phase_folders=("phase_folders",),
            )
        )
        self.assert_paths_as_expected(optimizer.paths)

    def test_combination_argument(self):
        optimizer = af.NonLinearOptimizer(
            "other",
            paths=af.Paths(
                phase_name="phase_name",
                phase_tag="phase_tag",
                phase_folders=("phase_folders",),
            ),
        )
        self.assert_paths_as_expected(optimizer.paths)
