import pytest

from autoconf import conf
import autofit as af

from test_autofit import mock

init_args = []
fit_args = []
fit_instances = []


class MockOptimizer(af.MockSearch):
    def __init__(self, paths=af.Paths()):
        super().__init__(paths=paths)
        init_args.append(paths.name)


class MockAnalysis(af.Analysis):
    prior_count = 2

    def log_likelihood_function(self, instance):
        fit_instances.append(instance)
        return 1

    def visualize(self, instance, during_analysis):
        pass

    def log(self, instance):
        pass


class MockClassContainer:
    def __init__(self):
        self.init_args = init_args
        self.fit_args = fit_args
        self.fit_instances = fit_instances

        self.MockOptimizer = MockOptimizer
        self.MockAnalysis = MockAnalysis


@pytest.fixture(name="container")
def make_mock_class_container():
    init_args.clear()
    fit_args.clear()
    fit_instances.clear()
    return MockClassContainer()
