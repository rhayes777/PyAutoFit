import os
import shutil

import pytest

from autofit import conf
from autofit import exc
from autofit import mock
from autofit.optimize import non_linear
from autofit.optimize.optimizer import grid


@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config(
        "{}/../../workspace/config".format(os.path.dirname(os.path.realpath(__file__))))


class MockAnalysis(non_linear.Analysis):
    def __init__(self):
        self.instances = []

    def fit(self, instance):
        self.instances.append(instance)
        try:
            return -((instance.one.redshift - 0.1) ** 2 + (instance.two.redshift - 0.7) ** 2)
        except AttributeError:
            return 0

    def visualize(self, instance, image_path, during_analysis):
        pass

    def describe(self, instance):
        return ""


def tuple_lists_equal(l1, l2):
    assert len(l1) == len(l2)
    for tuple_pair in zip(l1, l2):
        assert len(tuple_pair[0]) == len(tuple_pair[1])
        for item in zip(tuple_pair[0], tuple_pair[1]):
            if item[0] != pytest.approx(item[1]):
                return False
    return True


class TestGridSearchOptimizer(object):
    def test_config(self):
        assert non_linear.GridSearch(phase_name='').step_size == 0.1

    def test_1d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 1, 0.1)

        assert 10 == len(points)
        assert tuple_lists_equal(
            [(0.05,), (0.15,), (0.25,), (0.35,), (0.45,), (0.55,), (0.65,), (0.75,), (0.85,), (0.95,)],
            points)

    def test_2d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 2, 0.3)

        assert 9 == len(points)
        assert tuple_lists_equal([(0.15, 0.15), (0.15, 0.45), (0.15, 0.75),
                                  (0.45, 0.15), (0.45, 0.45), (0.45, 0.75),
                                  (0.75, 0.15), (0.75, 0.45), (0.75, 0.75),
                                  ],
                                 points)

    def test_3d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 3, 0.5)

        assert 3 == len(points[0])
        assert 8 == len(points)

    def test_best_fit(self):
        best_point = (0.65, 0.35)

        def fit(point):
            return -((best_point[0] - point[0]) ** 2 + (best_point[1] - point[1]) ** 2)

        result = grid(fit, 2, 0.1)

        assert pytest.approx(result[0]) == best_point[0]
        assert pytest.approx(result[1]) == best_point[1]


@pytest.fixture(name="grid_search")
def make_grid_search():
    name = "grid_search"
    try:
        shutil.rmtree("{}/{}/".format(conf.instance.output_path, name))
    except FileNotFoundError:
        pass
    return non_linear.GridSearch(phase_name=name, step_size=0.1)


class TestGridSearch(object):
    def test_1d(self, grid_search):
        grid_search.variable.one = mock.Galaxy

        analysis = MockAnalysis()
        grid_search.fit(analysis)

        assert len(analysis.instances) == 10

        instance = analysis.instances[5]

        assert isinstance(instance.one, mock.Galaxy)
        assert instance.one.redshift == 0.55

    def test_2d(self, grid_search):
        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis)

        assert pytest.approx(result.constant.one.redshift) == 0.05
        assert pytest.approx(result.constant.two.redshift) == 0.65

    def test_checkpoint_properties(self, grid_search):
        analysis = MockAnalysis()

        grid_search.variable.one = mock.Galaxy
        grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(phase_name="grid_search", step_size=0.1)

        assert grid_search.is_checkpoint
        assert grid_search.checkpoint_count == 10
        assert grid_search.checkpoint_fit == 0.
        assert grid_search.checkpoint_cube == (0.05,)
        assert grid_search.checkpoint_step_size == 0.1
        assert grid_search.checkpoint_prior_count == 1

    def test_recover_bad_checkpoint(self, grid_search):
        analysis = MockAnalysis()

        grid_search.variable.one = mock.Galaxy
        grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(phase_name="grid_search", step_size=0.1)

        with pytest.raises(exc.CheckpointException):
            grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(phase_name="grid_search", step_size=0.2)
        grid_search.variable.one = mock.Galaxy

        with pytest.raises(exc.CheckpointException):
            grid_search.fit(analysis)

    def test_recover_checkpoint(self, grid_search):
        analysis = MockAnalysis()

        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(phase_name="grid_search", step_size=0.1)

        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis)

        assert len(analysis.instances) == 1
        assert pytest.approx(result.constant.one.redshift) == 0.05
        assert pytest.approx(result.constant.two.redshift) == 0.65

    def test_recover_midway(self, grid_search):
        string = "11\n0\n(0.0, 0.0)\n0.1\n2"
        with open(grid_search.checkpoint_path, "w+") as f:
            f.write(string)

        grid_search = non_linear.GridSearch(phase_name="grid_search", step_size=0.1)

        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis)

        assert len(analysis.instances) == 90
        assert pytest.approx(result.constant.one.redshift) == 0.15
        assert pytest.approx(result.constant.two.redshift) == 0.65

    def test_instances(self, grid_search):
        grid_search.variable.one = mock.Galaxy

        analysis = MockAnalysis()
        result = grid_search.fit(analysis)

        assert len(result.instances) == 10

        instance = result.instances[5]

        assert isinstance(instance[0].one, mock.Galaxy)
        assert instance[0].one.redshift == 0.55
