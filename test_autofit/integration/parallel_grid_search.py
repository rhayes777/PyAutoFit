import logging
import shutil
from os import path

import test
import autofit as af
from autofit.optimize.grid_search import GridSearch

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


if __name__ == "__main__":
    grid_search = GridSearch(
        phase_name="phase_grid_search",
        phase_tag="_tag",
        phase_folders=["integration"],
        optimizer_class=af.MultiNest,
        parallel=False,
    )
    grid_search.model.profile = test_autofit.mock.EllipticalProfile

    # noinspection PyUnresolvedReferences
    result = grid_search.fit(
        Analysis(),
        [grid_search.model.profile.centre_0, grid_search.model.profile.centre_1],
    )

    print(result)
