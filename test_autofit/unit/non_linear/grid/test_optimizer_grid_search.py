import pickle

import pytest

import autofit as af
from autofit import exc
from autofit.mock import mock
from autofit.mock.mock import MockAnalysis


@pytest.fixture(name="mapper")
def make_mapper():
    mapper = af.ModelMapper()
    mapper.component = mock.MockClassx2Tuple
    return mapper


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    return af.NonLinearSearchGridSearch(
        af.Paths(name=""), number_of_steps=10, search=af.MockSearch()
    )


def test_unpickle_result():
    result = af.GridSearchResult(
        [af.Result(samples=None, previous_model=None)],
        lower_limit_lists=[[1]],
        physical_lower_limits_lists=[[1]],
    )
    result = pickle.loads(pickle.dumps(result))
    assert result is not None


class TestGridSearchablePriors:
    def test_generated_models(self, grid_search, mapper):

        mappers = list(
            grid_search.model_mappers(
                mapper,
                grid_priors=[
                    mapper.component.one_tuple.one_tuple_0,
                    mapper.component.one_tuple.one_tuple_1,
                ],
            )
        )

        assert len(mappers) == 100

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.1
        assert mappers[0].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_1.upper_limit == 0.2

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 0.9
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 1.0
        assert mappers[-1].component.one_tuple.one_tuple_1.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_1.upper_limit == 2.0

    def test_non_grid_searched_dimensions(self, mapper):
        grid_search = af.NonLinearSearchGridSearch(
            af.Paths(name=""), number_of_steps=10, search=af.MockSearch()
        )

        mappers = list(
            grid_search.model_mappers(
                mapper, grid_priors=[mapper.component.one_tuple.one_tuple_0]
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.1
        assert mappers[0].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_1.upper_limit == 2.0

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 0.9
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 1.0
        assert mappers[-1].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[-1].component.one_tuple.one_tuple_1.upper_limit == 2.0

    def test_tied_priors(self, grid_search, mapper):
        mapper.component.one_tuple.one_tuple_0 = mapper.component.one_tuple.one_tuple_1

        mappers = list(
            grid_search.model_mappers(
                grid_priors=[
                    mapper.component.one_tuple.one_tuple_0,
                    mapper.component.one_tuple.one_tuple_1,
                ],
                model=mapper,
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.2
        assert mappers[0].component.one_tuple.one_tuple_1.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_1.upper_limit == 0.2

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 2.0
        assert mappers[-1].component.one_tuple.one_tuple_1.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_1.upper_limit == 2.0

        for mapper in mappers:
            assert (
                mapper.component.one_tuple.one_tuple_0
                == mapper.component.one_tuple.one_tuple_1
            )

    def test_different_prior_width(self, grid_search, mapper):
        mapper.component.one_tuple.one_tuple_0 = af.UniformPrior(0.0, 2.0)
        mappers = list(
            grid_search.model_mappers(
                grid_priors=[mapper.component.one_tuple.one_tuple_0], model=mapper
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 0.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 0.2

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 1.8
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 2.0

        mapper.component.one_tuple.one_tuple_0 = af.UniformPrior(1.0, 1.5)
        mappers = list(
            grid_search.model_mappers(
                mapper, grid_priors=[mapper.component.one_tuple.one_tuple_0]
            )
        )

        assert len(mappers) == 10

        assert mappers[0].component.one_tuple.one_tuple_0.lower_limit == 1.0
        assert mappers[0].component.one_tuple.one_tuple_0.upper_limit == 1.05

        assert mappers[-1].component.one_tuple.one_tuple_0.lower_limit == 1.45
        assert mappers[-1].component.one_tuple.one_tuple_0.upper_limit == 1.5

    def test_raises_exception_for_bad_limits(self, grid_search, mapper):
        mapper.component.one_tuple.one_tuple_0 = af.GaussianPrior(
            0.0, 2.0, lower_limit=float("-inf"), upper_limit=float("inf")
        )
        with pytest.raises(exc.PriorException):
            list(
                grid_search.make_arguments(
                    [[0, 1]], grid_priors=[mapper.component.one_tuple.one_tuple_0]
                )
            )


@pytest.fixture(name="grid_search_05")
def make_grid_search_05():
    return af.NonLinearSearchGridSearch(
        search=MockOptimizer(), number_of_steps=2, paths=af.Paths(name="sample_name")
    )


class MockOptimizer(af.MockSearch):
    init_args = list()

    def __init__(self, paths=None):
        super().__init__(paths=paths or af.Paths(), fit_fast=False)
        self.init_args.append(paths)


@pytest.fixture(autouse=True)
def empty_args():
    MockOptimizer.init_args = list()


class TestGridNLOBehaviour:
    def test_results(self, grid_search_05, mapper):
        result = grid_search_05.fit(
            model=mapper,
            analysis=MockAnalysis(),
            grid_priors=[
                mapper.component.one_tuple.one_tuple_0,
                mapper.component.one_tuple.one_tuple_1,
            ],
        )

        assert len(result.results) == 4
        assert result.no_dimensions == 2

        grid_search = af.NonLinearSearchGridSearch(
            search=MockOptimizer(),
            number_of_steps=10,
            paths=af.Paths(name="sample_name"),
        )
        result = grid_search.fit(
            model=mapper,
            analysis=MockAnalysis(),
            grid_priors=[
                mapper.component.one_tuple.one_tuple_0,
                mapper.component.one_tuple.one_tuple_1,
            ],
        )

        assert len(result.results) == 100
        assert result.no_dimensions == 2
        assert result.max_log_likelihood_values.shape == (10, 10)

    # def test_results_parallel(self, mapper, container):
    #     grid_search = af.NonLinearSearchGridSearch(
    #         search=container.MockOptimizer,
    #         number_of_steps=10,
    #         paths=af.Paths(name="sample_name"),
    #         parallel=True,
    #     )
    #     result = grid_search.fit(
    #         container.MockAnalysis(),
    #         mapper,
    #         [mapper.component.one_tuple.one_tuple_0, mapper.component.one_tuple.one_tuple_1],
    #     )
    #
    #     assert len(result.results) == 100
    #     assert result.no_dimensions == 2
    #     assert result.likelihood_merit_array.shape == (10, 10)

    # def test_generated_models_with_instances(self, grid_search, container, mapper):
    #     instance_component = mock.MockClassx2Tuple()
    #     mapper.instance_component = instance_component
    #
    #     analysis = container.MockAnalysis()
    #
    #     grid_search.fit(analysis, mapper, [mapper.component.one_tuple.one_tuple_0])
    #
    #     for instance in container.fit_instances:
    #         assert isinstance(instance.component, mock.MockClassx2Tuple)
    #         assert instance.instance_component == instance_component
    #
    # def test_generated_models_with_instance_attributes(
    #         self, grid_search, mapper, container
    # ):
    #     instance = 2.0
    #     mapper.component.one_tuple.one_tuple_1 = instance
    #
    #     analysis = container.MockAnalysis()
    #
    #     grid_search.fit(analysis, mapper, [mapper.component.one_tuple.one_tuple_0])
    #
    #     assert len(container.fit_instances) > 0
    #
    #     for instance in container.fit_instances:
    #         assert isinstance(instance.component, mock.MockClassx2Tuple)
    #         # noinspection PyUnresolvedReferences
    #         assert instance.component.centre[1] == 2

    def test_passes_attributes(self):
        grid_search = af.NonLinearSearchGridSearch(
            af.Paths(name=""), number_of_steps=10, search=af.DynestyStatic()
        )

        grid_search.n_live_points = 20
        grid_search.sampling_efficiency = 0.3

        search = grid_search.search_instance("name_path")

        assert search.n_live_points is grid_search.n_live_points
        assert grid_search.paths.path != search.paths.path
        assert grid_search.paths.output_path != search.paths.output_path


class MockResult:
    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood
        self.model = log_likelihood


@pytest.fixture(name="grid_search_result")
def make_grid_search_result():
    one = MockResult(1)
    two = MockResult(2)

    # noinspection PyTypeChecker
    return af.GridSearchResult([one, two], [[1], [2]], [[1], [2]])


class TestGridSearchResult:
    def test_best_result(self, grid_search_result):
        assert grid_search_result.best_result.log_likelihood == 2

    def test_attributes(self, grid_search_result):
        assert grid_search_result.model == 2

    def test_best_model(self, grid_search_result):
        assert grid_search_result.best_model == 2

    def test_all_models(self, grid_search_result):
        assert grid_search_result.all_models == [1, 2]

    def test__result_derived_properties(self):
        lower_limit_lists = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]
        physical_lower_limits_lists = [
            [-2.0, -3.0],
            [-2.0, 0.0],
            [0.0, -3.0],
            [0.0, 0.0],
        ]

        grid_search_result = af.GridSearchResult(
            results=None,
            physical_lower_limits_lists=physical_lower_limits_lists,
            lower_limit_lists=lower_limit_lists,
        )

        print(grid_search_result)

        assert grid_search_result.shape == (2, 2)
        assert grid_search_result.physical_step_sizes == (2.0, 3.0)
        assert grid_search_result.physical_centres_lists == [
            [-1.0, -1.5],
            [-1.0, 1.5],
            [1.0, -1.5],
            [1.0, 1.5],
        ]
        assert grid_search_result.physical_upper_limits_lists == [
            [0.0, 0.0],
            [0.0, 3.0],
            [2.0, 0.0],
            [2.0, 3.0],
        ]
