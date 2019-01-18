import logging
import shutil
from os import path

import pytest

from autofit import mock
from autofit.optimize import grid_search as gs
from autofit.optimize import non_linear

logger = logging.getLogger(__name__)

try:
    output_dir = "{}/../../workspace/output/phase".format(path.dirname(path.realpath(__file__)))
    logger.info("Removing {}".format(output_dir))
    shutil.rmtree(output_dir)
except FileNotFoundError:
    logging.info("Not found")

try:
    output_dir = "{}/../../workspace/output/integration_grid_search".format(path.dirname(path.realpath(__file__)))
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


if __name__ == "__main__":
    TestCase().test_integration()
    TestCase().test_grid()
