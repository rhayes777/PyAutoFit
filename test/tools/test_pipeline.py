import os

import pytest

import autofit.optimize.non_linear.non_linear
import test.mock
from autofit import exc
from test import mock
from autofit.mapper import prior as p
from autofit.optimize import non_linear
from autofit.tools import phase as ph
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


class MockPhase(ph.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name, optimizer=None):
        super().__init__(phase_name)
        self.optimizer = optimizer
        self.phase_path = phase_name
        self.phase_tag = phase_name


class TestPipeline(object):
    def test_unique_phases(self):
        pipeline.Pipeline("name", MockPhase("one"), MockPhase("two"))
        with pytest.raises(exc.PipelineException):
            pipeline.Pipeline("name", MockPhase("one"), MockPhase("one"))

    def test_optimizer_assertion(self):
        optimizer = autofit.optimize.non_linear.non_linear.NonLinearOptimizer("Phase Name")
        optimizer.variable.profile = test.mock.GeometryProfile
        phase = MockPhase("phase_name", optimizer)

        try:
            os.makedirs(phase.make_path())
        except FileExistsError:
            pass

        phase.save_optimizer_for_phase()
        phase.assert_optimizer_pickle_matches_for_phase()

        optimizer.variable.profile.centre_0 = p.UniformPrior()

        with pytest.raises(exc.PipelineException):
            phase.assert_optimizer_pickle_matches_for_phase()

    def test_name_composition(self):
        first = pipeline.Pipeline("first")
        second = pipeline.Pipeline("second")

        assert (first + second).pipeline_name == "first + second"

    def test_assert_and_save_pickle(self):
        phase = ph.AbstractPhase("name")

        phase.assert_and_save_pickle()
        phase.assert_and_save_pickle()

        phase.variable.galaxy = test.mock.Galaxy

        with pytest.raises(exc.PipelineException):
            phase.assert_and_save_pickle()


# noinspection PyUnresolvedReferences
class TestPhasePipelineName(object):
    def test_name_stamping(self):
        one = MockPhase("one")
        two = MockPhase("two")
        pipeline.Pipeline("name", one, two)

        assert one.pipeline_name == "name"
        assert two.pipeline_name == "name"

    def test_no_restamping(self):
        one = MockPhase("one")
        two = MockPhase("two")
        pipeline_one = pipeline.Pipeline("one", one)
        pipeline_two = pipeline.Pipeline("two", two)

        composed_pipeline = pipeline_one + pipeline_two

        assert composed_pipeline[0].pipeline_name == "one"
        assert composed_pipeline[1].pipeline_name == "two"

        assert one.pipeline_name == "one"
        assert two.pipeline_name == "two"
