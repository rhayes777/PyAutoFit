import logging
import shutil
from os import path

import pytest


from autofit.optimize import grid_search as gs

import test

import autofit as af

logger = logging.getLogger(__name__)

try:
    output_dir = "{}/../../autolens_workspace/output".format(
        path.dirname(path.realpath(__file__))
    )
    logger.info("Removing {}".format(output_dir))
    shutil.rmtree(output_dir)
except FileNotFoundError:
    logging.info("Not found")


class Analysis(af.Analysis):
    def fit(self, instance):
        return -(instance.profile.centre[0] ** 2 + instance.profile.centre[1] ** 2)

    def visualize(self, instance, **kwargs):
        pass

    def describe(self, instance):
        return "{}, {}".format(*instance.profile.centre)


class Phase(af.AbstractPhase):
    profile = af.PhaseProperty("profile")
    instance_profile = af.PhaseProperty("instance_profile")

    def __init__(
        self,
        phase_name,
        phase_folders,
        profile,
        instance_profile,
        optimizer_class=af.MultiNest,
    ):
        super().__init__(
            phase_name=phase_name,
            phase_tag="phase_tag",
            phase_folders=phase_folders,
            optimizer_class=optimizer_class,
        )
        self.profile = profile
        self.instance_profile = instance_profile

    def make_result(self, result, analysis):
        return result


class TestCase(object):
    def test_integration(self):

        multinest = af.MultiNest(
            phase_folders=["integration"], phase_name="test_autoarray"
        )

        multinest.model.profile = test_autofit.mock.EllipticalProfile

        result = multinest.fit(Analysis())

        centre = result.instance.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_grid(self):

        grid_search = gs.GridSearch(
            phase_name="phase_grid_search",
            phase_tag="_tag",
            phase_folders=["integration"],
        )
        grid_search.model.profile = test_autofit.mock.EllipticalProfile

        # noinspection PyUnresolvedReferences
        result = grid_search.fit(
            Analysis(),
            [grid_search.model.profile.centre_0, grid_search.model.profile.centre_1],
        )

    def test_phase(self):

        phase = Phase(
            phase_name="test_phase",
            phase_folders=["integration"],
            profile=test_autofit.mock.EllipticalProfile,
            instance_profile=test_autofit.mock.EllipticalProfile(),
        )
        result = phase.run_analysis(Analysis())

        centre = result.instance.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_classic_grid_search_phase(self):
        # noinspection PyTypeChecker
        phase = Phase(
            phase_name="phase_classic_grid_search_phase",
            phase_folders=["integration"],
            profile=test_autofit.mock.EllipticalProfile,
            instance_profile=test_autofit.mock.EllipticalProfile(),
            optimizer_class=af.GridSearch,
        )
        result = phase.run_analysis(Analysis())

        centre = result.instance.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_grid_search_phase(self):
        class GridSearchPhase(af.as_grid_search(Phase)):
            @property
            def grid_priors(self):
                return [self.model.profile.centre_0, self.model.profile.centre_1]

        instance_profile = test_autofit.mock.EllipticalProfile()

        result = GridSearchPhase(
            phase_name="grid_search_phase",
            phase_folders=["integration"],
            number_of_steps=2,
            profile=test_autofit.mock.EllipticalProfile,
            instance_profile=instance_profile,
        ).run_analysis(Analysis())

        assert result.results[0].instance.instance_profile == instance_profile

        assert result.figure_of_merit_array[0, 0] > result.figure_of_merit_array[0, 1]
        assert result.figure_of_merit_array[0, 0] > result.figure_of_merit_array[1, 0]
        assert result.figure_of_merit_array[1, 0] > result.figure_of_merit_array[1, 1]
        assert result.figure_of_merit_array[0, 1] > result.figure_of_merit_array[1, 1]

    def test__grid_search_phase_parallel(self):
        class GridSearchPhase(af.as_grid_search(Phase, parallel=True)):
            @property
            def grid_priors(self):
                return [self.model.profile.centre_0, self.model.profile.centre_1]

        instance_profile = test_autofit.mock.EllipticalProfile()

        result = GridSearchPhase(
            phase_name="grid_search_phase_parallel",
            phase_folders=["integration"],
            number_of_steps=2,
            profile=test_autofit.mock.EllipticalProfile,
            instance_profile=instance_profile,
        ).run_analysis(Analysis())

        assert result.results[0].instance.instance_profile == instance_profile

        print(result.figure_of_merit_array)

        assert result.figure_of_merit_array[0, 0] > result.figure_of_merit_array[0, 1]
        assert result.figure_of_merit_array[0, 0] > result.figure_of_merit_array[1, 0]
        assert result.figure_of_merit_array[1, 0] > result.figure_of_merit_array[1, 1]
        assert result.figure_of_merit_array[0, 1] > result.figure_of_merit_array[1, 1]


if __name__ == "__main__":
    TestCase().test_integration()
    TestCase().test_grid()
    TestCase().test_phase()
    TestCase().test_classic_grid_search_phase()
    TestCase().test_grid_search_phase()
