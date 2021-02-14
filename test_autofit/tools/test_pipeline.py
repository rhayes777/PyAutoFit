import os

import pytest

import autofit as af
from autofit.mock import mock


@pytest.fixture(name="results")
def make_results_collection():
    results = af.ResultsCollection()

    results.add("first phase", "one")
    results.add("second phase", "two")

    return results


class TestResultsCollection:
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
        with pytest.raises(af.exc.PipelineException):
            results.from_phase("third phase")


class MockPhase(af.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, search):
        super().__init__(search=search)

    def save_metadata(self, *args, **kwargs):
        pass


class TestPipeline:
    def test_unique_phases(self):

        phase1 = MockPhase(search=af.MockSearch("one"))
        phase2 = MockPhase(search=af.MockSearch("two"))

        af.Pipeline("name", "", None, phase1, phase2)
        with pytest.raises(af.exc.PipelineException):
            af.Pipeline(
                "name",
                "",
                None,
                MockPhase(search=af.MockSearch("one")),
                MockPhase(search=af.MockSearch("one")),
            )

    def test_search_assertion(self, model):
        paths = af.Paths("Phase Name")
        search = af.MockSearch(paths)
        phase = MockPhase(search=search)
        phase.model.profile = mock.MockClassx2Tuple

        try:
            os.makedirs(phase.paths.make_path())
        except FileExistsError:
            pass

        phase.model.profile.centre_0 = af.UniformPrior()

    def test_name_composition(self):
        first = af.Pipeline("first", "", None)
        second = af.Pipeline("second", "", None)

        assert (first + second).pipeline_name == "first + second"


# noinspection PyUnresolvedReferences
class TestPhasePipelineName:
    def test_name_stamping(self):
        one = MockPhase(search=af.MockSearch("one"))
        two = MockPhase(search=af.MockSearch("two"))
        af.Pipeline("name", "", None, one, two)

        assert one.pipeline_name == "name"
        assert two.pipeline_name == "name"

    def test_no_restamping(self):
        one = MockPhase(search=af.MockSearch("one"))
        two = MockPhase(search=af.MockSearch("two"))
        pipeline_one = af.Pipeline("one", "", None, one)
        pipeline_two = af.Pipeline("two", "", None, two)

        composed_pipeline = pipeline_one + pipeline_two

        assert composed_pipeline[0].pipeline_name == "one"
        assert composed_pipeline[1].pipeline_name == "two"

        assert one.pipeline_name == "one"
        assert two.pipeline_name == "two"
