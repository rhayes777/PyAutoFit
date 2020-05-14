import pytest

import autofit as af

init_args = []
fit_args = []
fit_instances = []


class MockSamples:
    def __init__(self, log_likelihoods):
        self.log_likelihoods = log_likelihoods
        self.max_log_likelihood_instance = af.ModelInstance()


class MockOptimizer(af.NonLinearOptimizer):
    def __init__(self, paths):
        super().__init__(paths)
        init_args.append(paths.phase_name)

    def _fit(self, model, fitness_function):
        raise NotImplementedError()

    def _full_fit(self, model, analysis):
        fit_args.append(analysis)
        # noinspection PyTypeChecker
        return af.Result(
            MockSamples(
                [1.0]
            ),
            analysis.log_likelihood_function(None)
        )


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
