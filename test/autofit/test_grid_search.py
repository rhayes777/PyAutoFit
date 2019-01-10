from autofit.core import non_linear
from autofit.core import prior as p
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


class TestGridSearchOptimizer(object):
    def test_1d(self):
        points = []
        grid(lambda x: points.append(x), 1, 0.1)
        assert 10 == len(points)


# class TestGridSearch(object):
#     def test_1d(self):
#         grid_search = non_linear.GridSearch(step_size=0.1)
#         grid_search.variable.one = p.UniformPrior()
#         analysis = MockAnalysis()
#         grid_search.fit(analysis)
#         assert len(analysis.instances) == 10
