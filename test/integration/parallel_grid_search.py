import logging
import shutil
from os import path

import autofit.optimize.non_linear.multi_nest
import autofit.optimize.non_linear.non_linear
import test.mock
from test import mock
from autofit.optimize import grid_search as gs
from autofit.optimize import non_linear

logger = logging.getLogger(__name__)

try:
    output_dir = "{}/../../workspace/output".format(path.dirname(path.realpath(__file__)))
    logger.info("Removing {}".format(output_dir))
    shutil.rmtree(output_dir)
except FileNotFoundError:
    logging.info("Not found")


class Analysis(autofit.optimize.non_linear.non_linear.Analysis):
    def fit(self, instance):
        return -(instance.profile.centre[0] ** 2 + instance.profile.centre[1] ** 2)

    def visualize(self, instance, **kwargs):
        pass

    def describe(self, instance):
        return "{}, {}".format(*instance.profile.centre)


if __name__ == "__main__":
    grid_search = gs.GridSearch(phase_name="phase_grid_search", phase_tag='_tag', phase_folders=['integration'],
                                optimizer_class=autofit.optimize.non_linear.multi_nest.MultiNest, parallel=True)
    grid_search.variable.profile = test.mock.EllipticalProfile

    # noinspection PyUnresolvedReferences
    result = grid_search.fit(Analysis(),
                             [grid_search.variable.profile.centre_0, grid_search.variable.profile.centre_1])

    print(result)
