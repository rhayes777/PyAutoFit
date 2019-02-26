from autofit.tools import pipeline


class TestResultsCollection(object):
    def test_with_name(self):
        results = pipeline.ResultsCollection()

        results.add("first phase", "one")
        results.add("second phase", "two")

        assert results.from_phase("first phase") == "one"
        assert results.from_phase("second phase") == "two"
