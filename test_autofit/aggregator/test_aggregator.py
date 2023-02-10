def test_completed_aggregator(
        aggregator
):
    aggregator = aggregator(
        aggregator.search.is_complete
    )
    assert len(aggregator) == 1


class TestLoading:
    def test_unzip(self, aggregator):
        assert len(aggregator) == 2

    def test_pickles(self, aggregator):
        assert list(aggregator.values("dataset"))[0]["name"] == "dataset"


class TestOperations:
    def test_attribute(self, aggregator):
        assert list(aggregator.values("pipeline")) == [
            "pipeline0",
            "pipeline1"
        ]

    def test_indexing(self, aggregator):
        assert list(aggregator[1:].values("pipeline")) == ["pipeline1"]
        assert list(aggregator[:1].values("pipeline")) == ["pipeline0"]
        assert list(aggregator[1: 2].values("pipeline")) == ["pipeline1"]
        assert list(aggregator[0: 1].values("pipeline")) == ["pipeline0"]
        assert list(aggregator[-1:].values("pipeline")) == ["pipeline1"]
        assert list(aggregator[:-1].values("pipeline")) == ["pipeline0"]
        assert aggregator[0]["pipeline"] == "pipeline0"
        assert aggregator[-1]["pipeline"] == "pipeline1"

    def test_map(self, aggregator):
        def some_function(fit):
            return f"{fit.id} {fit['pipeline']}"

        results = aggregator.map(some_function)
        assert list(results) == [
            'complete pipeline0',
            'incomplete pipeline1'
        ]
