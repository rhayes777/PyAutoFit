import os

import pytest

from autofit import exc
from autofit import mock
from autofit.mapper import prior as p
from autofit.optimize import non_linear
from autofit.tools import pipeline


@pytest.fixture(name="results")
def make_results_collection():
    results = pipeline.ResultsCollection()

    results.add("first phase", "one")
    results.add("second phase", "two")

    return results


class TestResultsCollection(object):
    def test_with_name(self, results):
        assert results.from_phase("first phase") == "one"
        assert results.from_phase("second phase") == "two"

    def test_with_index(self, results):
        assert results[0] == "one"
        assert results[1] == "two"
        assert results.first == "one"
        assert results.last == "two"
        assert len(results) == 2

    def test_missing_result(self, results):
        with pytest.raises(exc.PipelineException):
            results.from_phase("third phase")

    def test_duplicate_result(self, results):
        with pytest.raises(exc.PipelineException):
            results.add("second phase", "three")


class MockPhase(object):
    def __init__(self, phase_name, optimizer=None):
        self.phase_name = phase_name
        self.optimizer = optimizer
        self.phase_path = phase_name
        self.phase_tag = phase_name


class TestPipeline(object):
    def test_unique_phases(self):
        pipeline.Pipeline("name", MockPhase("one"), MockPhase("two"))
        with pytest.raises(exc.PipelineException):
            pipeline.Pipeline("name", MockPhase("one"), MockPhase("one"))

    def test_optimizer_assertion(self):
        optimizer = non_linear.NonLinearOptimizer("Phase Name")
        optimizer.variable.profile = mock.GeometryProfile
        phase = MockPhase("phase_name", optimizer)

        try:
            os.makedirs(pipeline.make_path(phase))
        except FileExistsError:
            pass

        pipeline.save_optimizer_for_phase(phase)
        pipeline.assert_optimizer_pickle_matches_for_phase(phase)

        optimizer.variable.profile.centre_0 = p.UniformPrior()

        with pytest.raises(exc.PipelineException):
            pipeline.assert_optimizer_pickle_matches_for_phase(phase)

    def test_name_composition(self):
        first = pipeline.Pipeline("first")
        second = pipeline.Pipeline("second")

        assert (first + second).pipeline_name == "first + second"
