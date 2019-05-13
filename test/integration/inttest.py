import logging
import shutil
from os import path

import pytest

from autofit import mock
from autofit.tools import phase as p
from autofit.tools import phase_property
from autofit.optimize import grid_search as gs
from autofit.optimize import non_linear

logger = logging.getLogger(__name__)

try:
    output_dir = "{}/../../workspace/output".format(path.dirname(path.realpath(__file__)))
    logger.info("Removing {}".format(output_dir))
    shutil.rmtree(output_dir)
except FileNotFoundError:
    logging.info("Not found")


class Analysis(non_linear.Analysis):
    def fit(self, instance):
        return -(instance.profile.centre[0] ** 2 + instance.profile.centre[1] ** 2)

    def visualize(self, instance, **kwargs):
        pass

    def describe(self, instance):
        return "{}, {}".format(*instance.profile.centre)


class Phase(p.AbstractPhase):
    profile = phase_property.PhaseProperty("profile")
    constant_profile = phase_property.PhaseProperty("constant_profile")

    def __init__(self, phase_name, tag_phases, phase_folders, profile, constant_profile,
                 optimizer_class=non_linear.MultiNest):
        super().__init__(phase_name=phase_name, tag_phases=tag_phases, phase_tag='_tag', phase_folders=phase_folders,
                         optimizer_class=optimizer_class)
        self.profile = profile
        self.constant_profile = constant_profile

    def make_result(self, result, analysis):
        return result


class TestCase(object):
    
    def test_integration(self):
        multinest = non_linear.MultiNest(phase_folders=['integration'], phase_name='test')

        multinest.variable.profile = mock.EllipticalProfile

        result = multinest.fit(Analysis())

        centre = result.constant.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_grid(self):

        grid_search = gs.GridSearch(phase_name="phase_grid_search", phase_tag='_tag', phase_folders=['integration'])
        grid_search.variable.profile = mock.EllipticalProfile

        # noinspection PyUnresolvedReferences
        result = grid_search.fit(Analysis(),
                                 [grid_search.variable.profile.centre_0, grid_search.variable.profile.centre_1])

    def test_phase(self):
        
        phase = Phase(phase_name="test_phase", tag_phases=True, phase_folders=['integration'],
                       profile=mock.EllipticalProfile, constant_profile=mock.EllipticalProfile())
        result = phase.run_analysis(Analysis())

        centre = result.constant.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_classic_grid_search_phase(self):
        # noinspection PyTypeChecker
        phase = Phase(phase_name="phase_classic_grid_search_phase",  tag_phases=True, phase_folders=['integration'],
                      profile=mock.EllipticalProfile, constant_profile=mock.EllipticalProfile(),
                      optimizer_class=non_linear.GridSearch)
        result = phase.run_analysis(Analysis())

        centre = result.constant.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_grid_search_phase(self):

        class GridSearchPhase(p.as_grid_search(Phase)):
            @property
            def grid_priors(self):
                return [self.variable.profile.centre_0, self.variable.profile.centre_1]

        constant_profile = mock.EllipticalProfile()

        result = GridSearchPhase(phase_name="grid_search_phase", tag_phases=True, phase_folders=['integration'],
                                 number_of_steps=2, profile=mock.EllipticalProfile,
                                 constant_profile=constant_profile).run_analysis(Analysis())

        assert result.results[0].constant.constant_profile == constant_profile

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
