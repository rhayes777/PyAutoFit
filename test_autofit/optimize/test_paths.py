import autofit as af


class TestPathDecorator:
    def test_with_arguments(self):
        optimizer = af.NonLinearOptimizer(
            phase_name="phase_name",
            phase_tag="phase_tag",
            phase_folders=("phase_folders",),
        )
        paths = optimizer.paths

        assert paths.phase_name == "phase_name"
        assert paths.phase_tag == "phase_tag"
        assert paths.phase_folders == ["phase_folders"]
