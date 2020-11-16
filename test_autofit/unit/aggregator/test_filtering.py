from os import path

class TestFiltering:
    def test_or(self, aggregator):
        predicate_one = aggregator.directory.contains("one")
        predicate_two = aggregator.directory.contains("two")
        result = aggregator.filter(predicate_one | predicate_two)
        assert len(result) == 2
        assert result.directories == [path.join("directory", "number", "one"), path.join("directory", "number", "two")]

    def test_and(self, aggregator):
        predicate_one = aggregator.pipeline == "pipeline1"
        predicate_two = aggregator.phase == "phase2"
        result = aggregator.filter(predicate_one & predicate_two)
        assert len(result) == 1
        assert result.directories == [path.join("directory", "number", "two")]

    def test_not_contains(self, aggregator):
        predicate = ~(aggregator.pipeline.contains("1"))
        result = aggregator.filter(predicate)
        assert len(result) == 1
        assert result[0].pipeline == "pipeline2"

    def test_not_equal(self, aggregator):
        predicate = aggregator.pipeline != "pipeline1"
        result = aggregator.filter(predicate)
        assert len(result) == 1
        assert result[0].pipeline == "pipeline2"

    def test_rhs(self, aggregator):
        predicate = "pipeline1" != aggregator.pipeline
        result = aggregator.filter(predicate)
        assert result.pipeline == ["pipeline2"]

        predicate = "pipeline1" == aggregator.pipeline
        result = aggregator.filter(predicate)
        assert result.pipeline == ["pipeline1", "pipeline1"]

    def test_filter_index(self, aggregator):
        assert list(
            aggregator.filter(aggregator.pipeline == "pipeline1")[1:].values("pipeline")
        ) == ["pipeline1"]
        assert list(
            aggregator[1:].filter(aggregator.pipeline == "pipeline1").values("pipeline")
        ) == ["pipeline1"]

    def test_filter(self, aggregator):
        result = aggregator.filter(aggregator.pipeline == "pipeline1")
        assert len(result) == 2
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(
            aggregator.pipeline == "pipeline1", aggregator.phase == "phase1"
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(aggregator.pipeline == "pipeline1").filter(
            aggregator.phase == "phase1"
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

    def test_filter_contains(self, aggregator):
        result = aggregator.filter(aggregator.pipeline.contains("1"))
        assert len(result) == 2
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(
            aggregator.pipeline.contains("1"), aggregator.phase.contains("1")
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

        result = aggregator.filter(aggregator.pipeline.contains("1")).filter(
            aggregator.phase == "phase1"
        )

        assert len(result) == 1
        assert result[0].pipeline == "pipeline1"

    def test_filter_contains_directory(self, aggregator):
        result = aggregator.filter(aggregator.directory.contains("number"))
        assert len(result) == 2

        result = aggregator.filter(aggregator.directory.contains("letter"))
        assert len(result) == 1
