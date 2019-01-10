import shutil

import pytest

from autofit import conf
from autofit import exc
from autofit import mock
from autofit.core import non_linear
from autofit.core.optimizer import grid


class MockAnalysis(non_linear.Analysis):
    def __init__(self):
        self.instances = []

    def fit(self, instance):
        self.instances.append(instance)
        try:
            return 1 if pytest.approx(instance.one.redshift) == 0.1 and pytest.approx(
                instance.two.redshift) == 0.7 else 0
        except AttributeError:
            return 0

    def visualize(self, instance, suffix, during_analysis):
        pass

    def log(self, instance):
        pass


def tuple_lists_equal(l1, l2):
    assert len(l1) == len(l2)
    for tuple_pair in zip(l1, l2):
        assert len(tuple_pair[0]) == len(tuple_pair[1])
        for item in zip(tuple_pair[0], tuple_pair[1]):
            if pytest.approx(item[0]) != pytest.approx(item[1]):
                return False
    return True


class TestGridSearchOptimizer(object):
    def test_1d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 1, 0.1)

        assert 11 == len(points)
        assert tuple_lists_equal(
            [(0.0,), (0.1,), (0.2,), (0.3,), (0.4,), (0.5,), (0.6,), (0.7,), (0.8,), (0.9,), (1.0,)],
            points)

    def test_2d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 2, 0.3)

        assert 16 == len(points)
        assert tuple_lists_equal([(0.0, 0.0), (0.0, 0.3), (0.0, 0.6), (0.0, 0.9),
                                  (0.3, 0.0), (0.3, 0.3), (0.3, 0.6), (0.3, 0.9),
                                  (0.6, 0.0), (0.6, 0.3), (0.6, 0.6), (0.6, 0.9),
                                  (0.9, 0.0), (0.9, 0.3), (0.9, 0.6), (0.9, 0.9), ],
                                 points)

    def test_3d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 3, 0.5)

        assert 3 == len(points[0])
        assert 27 == len(points)

    def test_best_fit(self):
        best_point = (0.6, 0.3)

        def fit(point):
            return 1 if point == best_point else 0

        result = grid(fit, 2, 0.3)

        assert result == best_point


@pytest.fixture(name="grid_search")
def make_grid_search():
    name = "grid_search"
    try:
        shutil.rmtree("{}/{}/".format(conf.instance.output_path, name))
    except FileNotFoundError:
        pass
    return non_linear.GridSearch(name=name, step_size=0.1)


class TestGridSearch(object):
    def test_1d(self, grid_search):
        grid_search.variable.one = mock.Galaxy

        analysis = MockAnalysis()
        grid_search.fit(analysis)

        assert len(analysis.instances) == 11

        instance = analysis.instances[5]

        assert isinstance(instance.one, mock.Galaxy)
        assert instance.one.redshift == 0.5

    def test_2d(self, grid_search):
        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis)

        assert pytest.approx(result.constant.one.redshift) == 0.1
        assert pytest.approx(result.constant.two.redshift) == 0.7

    def test_checkpoint_properties(self, grid_search):
        analysis = MockAnalysis()

        grid_search.variable.one = mock.Galaxy
        grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(name="grid_search", step_size=0.1)

        assert grid_search.is_checkpoint
        assert grid_search.checkpoint_count == 11
        assert grid_search.checkpoint_fit == 0.
        assert grid_search.checkpoint_cube == (0.0,)
        assert grid_search.checkpoint_step_size == 0.1
        assert grid_search.checkpoint_prior_count == 1

    def test_recover_bad_checkpoint(self, grid_search):
        analysis = MockAnalysis()

        grid_search.variable.one = mock.Galaxy
        grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(name="grid_search", step_size=0.1)

        with pytest.raises(exc.CheckpointException):
            grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(name="grid_search", step_size=0.2)
        grid_search.variable.one = mock.Galaxy

        with pytest.raises(exc.CheckpointException):
            grid_search.fit(analysis)

    def test_recover_checkpoint(self, grid_search):
        analysis = MockAnalysis()

        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        grid_search.fit(analysis)

        grid_search = non_linear.GridSearch(name="grid_search", step_size=0.1)

        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis)

        assert len(analysis.instances) == 1
        assert pytest.approx(result.constant.one.redshift) == 0.1
        assert pytest.approx(result.constant.two.redshift) == 0.7

    def test_recover_midway(self, grid_search):
        string = "11\n0\n(0.0, 0.0)\n0.1\n2"
        with open(grid_search.checkpoint_path, "w+") as f:
            f.write(string)

        grid_search = non_linear.GridSearch(name="grid_search", step_size=0.1)

        grid_search.variable.one = mock.Galaxy
        grid_search.variable.two = mock.Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis)

        assert len(analysis.instances) == 111
        assert pytest.approx(result.constant.one.redshift) == 0.1
        assert pytest.approx(result.constant.two.redshift) == 0.7
