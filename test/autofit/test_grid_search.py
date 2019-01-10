from autofit.core import prior as p
from autofit.core import non_linear


class MockAnalysis(non_linear.Analysis):
    def __init__(self):
        self.instances = []

    def fit(self, instance):
        self.instances.append(instance)

    def visualize(self, instance, suffix, during_analysis):
        pass

    def log(self, instance):
        pass


class TestGridSearch(object):
    def test_1d(self):
        grid_search = non_linear.GridSearch(step_size=0.1)
        grid_search.variable.one = p.UniformPrior()
        analysis = MockAnalysis()
        grid_search.fit(analysis)
        assert len(analysis.instances) == 11
