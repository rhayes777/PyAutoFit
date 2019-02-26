import pytest

from autofit import exc
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
    def __init__(self, phase_name):
        self.phase_name = phase_name


class TestPipeline(object):
    def test_unique_phases(self):
        pipeline.Pipeline("name", MockPhase("one"), MockPhase("two"))
        with pytest.raises(exc.PipelineException):
            pipeline.Pipeline("name", MockPhase("one"), MockPhase("one"))
