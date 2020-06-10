import pickle

import numpy as np
import pytest

import autofit as af
from autofit import exc, DownhillSimplex
from test_autofit.unit.mapper.model.test_model_mapper import GeometryProfile


@pytest.fixture(name="mapper")
def make_mapper():
    mapper = af.ModelMapper()
    mapper.profile = GeometryProfile
    return mapper


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    return af.NonLinearSearchGridSearch(af.Paths(name=""), number_of_steps=10, non_linear_class=DownhillSimplex)


def test_unpickle_result():
    result = af.GridSearchResult(
        [af.Result(
            samples=None
        )],
        lower_limit_lists=[[1]],
        physical_lower_limits_lists=[[1]]
    )
    result = pickle.loads(
        pickle.dumps(
            result
        )
    )
    assert result is not None


class TestGridSearchablePriors:
    def test_generated_models(self, grid_search, mapper):
        mappers = list(
            grid_search.model_mappers(
                mapper, grid_priors=[mapper.profile.centre_0, mapper.profile.centre_1]
            )
        )

        assert len(mappers) == 100

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 0.1

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.9
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

    def test_non_grid_searched_dimensions(self, mapper):
        grid_search = af.NonLinearSearchGridSearch(af.Paths(name=""), number_of_steps=10)

        mappers = list(
            grid_search.model_mappers(mapper, grid_priors=[mapper.profile.centre_0])
        )

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 1.0

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.0
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

    def test_tied_priors(self, grid_search, mapper):
        mapper.profile.centre_0 = mapper.profile.centre_1

        mappers = list(
            grid_search.model_mappers(
                grid_priors=[mapper.profile.centre_0, mapper.profile.centre_1],
                model=mapper,
            )
        )

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 0.1

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.9
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

        for mapper in mappers:
            assert mapper.profile.centre_0 == mapper.profile.centre_1

    def test_different_prior_width(self, grid_search, mapper):
        mapper.profile.centre_0 = af.UniformPrior(0.0, 2.0)
        mappers = list(
            grid_search.model_mappers(
                grid_priors=[mapper.profile.centre_0], model=mapper
            )
        )

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.2

        assert mappers[-1].profile.centre_0.lower_limit == 1.8
        assert mappers[-1].profile.centre_0.upper_limit == 2.0

        mapper.profile.centre_0 = af.UniformPrior(1.0, 1.5)
        mappers = list(
            grid_search.model_mappers(mapper, grid_priors=[mapper.profile.centre_0])
        )

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 1.0
        assert mappers[0].profile.centre_0.upper_limit == 1.05

        assert mappers[-1].profile.centre_0.lower_limit == 1.45
        assert mappers[-1].profile.centre_0.upper_limit == 1.5

    def test_raises_exception_for_bad_limits(self, grid_search, mapper):
        mapper.profile.centre_0 = af.GaussianPrior(
            0.0, 2.0, lower_limit=float("-inf"), upper_limit=float("inf")
        )
        with pytest.raises(exc.PriorException):
            list(
                grid_search.make_arguments(
                    [[0, 1]], grid_priors=[mapper.profile.centre_0]
                )
            )


@pytest.fixture(name="grid_search_05")
def make_grid_search_05(container):
    return af.NonLinearSearchGridSearch(
        non_linear_class=container.MockOptimizer,
        number_of_steps=2,
        paths=af.Paths(name="sample_name"),
    )


class TestGridNLOBehaviour:
    def test_calls(self, grid_search_05, container, mapper):
        result = grid_search_05.fit(
            model=mapper,
            analysis=container.MockAnalysis(),
            grid_priors=[mapper.profile.centre_0]
        )

        assert len(container.init_args) == 2
        assert len(container.fit_args) == 2
        assert len(result.results) == 2

    def test_names_1d(self, grid_search_05, container, mapper):
        grid_search_05.fit(
            model=mapper,
            analysis=container.MockAnalysis(),
            grid_priors=[mapper.profile.centre_0]
        )

        assert len(container.init_args) == 2
        print(container.init_args[0])
        assert container.init_args[0] == "sample_name///profile_centre_0_0.00_0.50"
        assert container.init_args[1] == "sample_name///profile_centre_0_0.50_1.00"

    def test_round_names(self, container, mapper):
        grid_search = af.NonLinearSearchGridSearch(
            non_linear_class=container.MockOptimizer,
            number_of_steps=3,
            paths=af.Paths(name="sample_name"),
        )

        grid_search.fit(model=mapper, analysis=container.MockAnalysis(), grid_priors=[mapper.profile.centre_0])

        assert len(container.init_args) == 3
        assert container.init_args[0] == "sample_name///profile_centre_0_0.00_0.33"
        assert container.init_args[1] == "sample_name///profile_centre_0_0.33_0.67"
        assert container.init_args[2] == "sample_name///profile_centre_0_0.67_1.00"

    def test_names_2d(self, grid_search_05, mapper, container):
        grid_search_05.fit(
            model=mapper,
            analysis=container.MockAnalysis(),
            grid_priors=[mapper.profile.centre_0, mapper.profile.centre_1],
        )

        assert len(container.init_args) == 4

        sorted_args = list(sorted(container.init_args[n] for n in range(4)))

        assert (
                sorted_args[0]
                == "sample_name///profile_centre_0_0.00_0.50_profile_centre_1_0.00_0.50"
        )
        assert (
                sorted_args[1]
                == "sample_name///profile_centre_0_0.00_0.50_profile_centre_1_0.50_1.00"
        )
        assert (
                sorted_args[2]
                == "sample_name///profile_centre_0_0.50_1.00_profile_centre_1_0.00_0.50"
        )
        assert (
                sorted_args[3]
                == "sample_name///profile_centre_0_0.50_1.00_profile_centre_1_0.50_1.00"
        )

    def test_results(self, grid_search_05, mapper, container):
        result = grid_search_05.fit(
            model=mapper,
            analysis=container.MockAnalysis(),
            grid_priors=[mapper.profile.centre_0, mapper.profile.centre_1],
        )

        assert len(result.results) == 4
        assert result.no_dimensions == 2
        assert np.equal(
            result.max_log_likelihood_values, np.array([[1.0, 1.0], [1.0, 1.0]])
        ).all()

        grid_search = af.NonLinearSearchGridSearch(
            non_linear_class=container.MockOptimizer,
            number_of_steps=10,
            paths=af.Paths(name="sample_name"),
        )
        result = grid_search.fit(
            model=mapper,
            analysis=container.MockAnalysis(),
            grid_priors=[mapper.profile.centre_0, mapper.profile.centre_1],
        )

        assert len(result.results) == 100
        assert result.no_dimensions == 2
        assert result.max_log_likelihood_values.shape == (10, 10)

    # def test_results_parallel(self, mapper, container):
    #     grid_search = af.NonLinearSearchGridSearch(
    #         non_linear_class=container.MockOptimizer,
    #         number_of_steps=10,
    #         paths=af.Paths(phase_name="sample_name"),
    #         parallel=True,
    #     )
    #     result = grid_search.fit(
    #         container.MockAnalysis(),
    #         mapper,
    #         [mapper.profile.centre_0, mapper.profile.centre_1],
    #     )
    #
    #     assert len(result.results) == 100
    #     assert result.no_dimensions == 2
    #     assert result.likelihood_merit_array.shape == (10, 10)

    # def test_generated_models_with_instances(self, grid_search, container, mapper):
    #     instance_profile = GeometryProfile()
    #     mapper.instance_profile = instance_profile
    #
    #     analysis = container.MockAnalysis()
    #
    #     grid_search.fit(analysis, mapper, [mapper.profile.centre_0])
    #
    #     for instance in container.fit_instances:
    #         assert isinstance(instance.profile, GeometryProfile)
    #         assert instance.instance_profile == instance_profile
    #
    # def test_generated_models_with_instance_attributes(
    #         self, grid_search, mapper, container
    # ):
    #     instance = 2.0
    #     mapper.profile.centre_1 = instance
    #
    #     analysis = container.MockAnalysis()
    #
    #     grid_search.fit(analysis, mapper, [mapper.profile.centre_0])
    #
    #     assert len(container.fit_instances) > 0
    #
    #     for instance in container.fit_instances:
    #         assert isinstance(instance.profile, GeometryProfile)
    #         # noinspection PyUnresolvedReferences
    #         assert instance.profile.centre[1] == 2

    def test_passes_attributes(self):
        grid_search = af.NonLinearSearchGridSearch(
            af.Paths(name=""),
            number_of_steps=10,
            non_linear_class=af.MultiNest,
        )

        grid_search.n_live_points = 20
        grid_search.sampling_efficiency = 0.3

        search = grid_search.search_instance("name_path")

        assert search.n_live_points is grid_search.n_live_points
        assert search.sampling_efficiency is grid_search.sampling_efficiency
        assert grid_search.paths.path != search.paths.path
        assert grid_search.paths.backup_path != search.paths.backup_path
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
        physical_lower_limits_lists = [[-2.0, -3.0], [-2.0, 0.0], [0.0, -3.0], [0.0, 0.0]]

        grid_search_result = af.GridSearchResult(
            results=None,
            physical_lower_limits_lists=physical_lower_limits_lists,
            lower_limit_lists=lower_limit_lists
        )

        print(grid_search_result)

        assert grid_search_result.shape == (2, 2)
        assert grid_search_result.physical_step_sizes == (2.0, 3.0)
        assert grid_search_result.physical_centres_lists == [[-1.0, -1.5], [-1.0, 1.5], [1.0, -1.5], [1.0, 1.5]]
        assert grid_search_result.physical_upper_limits_lists == [[0.0, 0.0], [0.0, 3.0], [2.0, 0.0], [2.0, 3.0]]