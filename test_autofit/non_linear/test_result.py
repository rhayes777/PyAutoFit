import pytest
import autofit as af

@pytest.fixture(name="result")
def make_result():
    mapper = af.ModelMapper()
    mapper.component = af.m.MockClassx2Tuple
    # noinspection PyTypeChecker
    return af.Result(
        samples=af.m.MockSamples(gaussian_tuples=[(0, 0), (1, 0)]),
        model=mapper,
        search=af.m.MockSearch(),
    )


class TestResult:
    def test_model(self, result):
        component = result.model.component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.2
        assert component.one_tuple.one_tuple_1.sigma == 0.2

    def test_model_absolute(self, result):
        component = result.model_absolute(a=2.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 2.0
        assert component.one_tuple.one_tuple_1.sigma == 2.0

    def test_model_relative(self, result):
        component = result.model_relative(r=1.0).component
        assert component.one_tuple.one_tuple_0.mean == 0
        assert component.one_tuple.one_tuple_1.mean == 1
        assert component.one_tuple.one_tuple_0.sigma == 0.0
        assert component.one_tuple.one_tuple_1.sigma == 1.0

    def test_raises(self, result):
        with pytest.raises(af.exc.PriorException):
            result.model.mapper_from_gaussian_tuples(
                result.samples._gaussian_tuples, a=2.0, r=1.0
            )


@pytest.fixture(name="results")
def make_results_collection():
    results = af.ResultsCollection()

    results.add("first", "one")
    results.add("second", "two")

    return results


class TestResultsCollection:
    def test_with_name(self, results):
        assert results.from_name("first") == "one"
        assert results.from_name("second") == "two"

    def test_with_index(self, results):
        assert results[0] == "one"
        assert results[1] == "two"
        assert results.first == "one"
        assert results.last == "two"
        assert len(results) == 2

    def test_missing_result(self, results):
        with pytest.raises(af.exc.PipelineException):
            results.from_name("third")
