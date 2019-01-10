import pytest

from autofit import mock
from autofit.core import non_linear
from autofit.core.optimizer import grid


class MockAnalysis(non_linear.Analysis):
    def __init__(self, best_fit=(0.5,)):
        self.instances = []
        self.best_fit = best_fit

    def fit(self, instance):
        self.instances.append(instance)
        return 1 if instance == self.best_fit else 0

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


class TestGridSearch(object):
    def test_1d(self):
        grid_search = non_linear.GridSearch(step_size=0.1)
        grid_search.variable.one = mock.Galaxy

        analysis = MockAnalysis()
        grid_search.fit(analysis)

        assert len(analysis.instances) == 11
