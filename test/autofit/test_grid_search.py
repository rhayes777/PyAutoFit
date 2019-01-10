import pytest

from autofit.core import non_linear
from autofit.core.optimizer import grid


class MockAnalysis(non_linear.Analysis):
    def __init__(self):
        self.instances = []

    def fit(self, instance):
        self.instances.append(instance)

    def visualize(self, instance, suffix, during_analysis):
        pass

    def log(self, instance):
        pass


def tuple_lists_equal(l1, l2):
    for tuple_pair in zip(l1, l2):
        for item in zip(tuple_pair[0], tuple_pair[1]):
            if pytest.approx(item[0]) != pytest.approx(item[1]):
                return False
    return True


class TestGridSearchOptimizer(object):
    def test_1d(self):
        points = []
        grid(lambda x: points.append(x), 1, 0.1)
        assert 10 == len(points)
        assert tuple_lists_equal([(0.0,), (0.1,), (0.2,), (0.3,), (0.4,), (0.5,), (0.6,), (0.7,), (0.8,), (0.9,)],
                                 points)

# class TestGridSearch(object):
#     def test_1d(self):
#         grid_search = non_linear.GridSearch(step_size=0.1)
#         grid_search.variable.one = p.UniformPrior()
#         analysis = MockAnalysis()
#         grid_search.fit(analysis)
#         assert len(analysis.instances) == 10
