import logging
import shutil
from os import path

import pytest

from autofit import mock
from autofit.core import phase as p
from autofit.core import phase_property
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

    def log(self, instance):
        logger.info("{}, {}".format(*instance.profile.centre))


class Phase(p.AbstractPhase):
    profile = phase_property.PhaseProperty("profile")

    def __init__(self, phase_name, profile, optimizer_class=non_linear.MultiNest):
        super().__init__(phase_name=phase_name, optimizer_class=optimizer_class)
        self.profile = profile

    def make_result(self, result, analysis):
        return result


class TestCase(object):
    def test_integration(self):
        multinest = non_linear.MultiNest()

        multinest.variable.profile = mock.EllipticalProfile

        result = multinest.fit(Analysis())

        centre = result.constant.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_grid(self):
        grid_search = gs.GridSearch(name="integration_grid_search")
        grid_search.variable.profile = mock.EllipticalProfile

        # noinspection PyUnresolvedReferences
        result = grid_search.fit(Analysis(),
                                 [grid_search.variable.profile.centre_0, grid_search.variable.profile.centre_1])

        print(result.figure_of_merit_array)

    def test_phase(self):
        phase = Phase(phase_name="test_phase", profile=mock.EllipticalProfile)
        result = phase.run_analysis(Analysis())

        centre = result.constant.profile.centre

        assert 0 == pytest.approx(centre[0], abs=0.1)
        assert 0 == pytest.approx(centre[1], abs=0.1)

    def test_grid_search_phase(self):
        class GridSearchPhase(p.as_grid_search(Phase)):
            @property
            def grid_priors(self):
                return [self.variable.profile.centre_0, self.variable.profile.centre_1]

        result = GridSearchPhase(number_of_steps=2, phase_name="grid_search_phase",
                                 profile=mock.EllipticalProfile).run_analysis(Analysis())

        print(result.figure_of_merit_array)


if __name__ == "__main__":
    # TestCase().test_integration()
    # TestCase().test_grid()
    # TestCase().test_phase()
    TestCase().test_grid_search_phase()
