import numpy as np
import pytest

import autofit.optimize.non_linear.multi_nest
import autofit.optimize.non_linear.non_linear
from autofit import exc, Paths
from autofit.mapper import model_mapper as mm
from autofit.mapper.prior_model import prior as p
from autofit.optimize import grid_search as gs
from autofit.tools import phase
from test_autofit.mapper.test_model_mapper import GeometryProfile


@pytest.fixture(name="mapper")
def make_mapper():
    mapper = mm.ModelMapper()
    mapper.profile = GeometryProfile
    return mapper


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    return gs.GridSearch(Paths(phase_name=""), number_of_steps=10)


class TestGridSearchablePriors(object):
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
        grid_search = gs.GridSearch(Paths(phase_name=""), number_of_steps=10)

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
        mapper.profile.centre_0 = p.UniformPrior(0.0, 2.0)
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

        mapper.profile.centre_0 = p.UniformPrior(1.0, 1.5)
        mappers = list(
            grid_search.model_mappers(mapper, grid_priors=[mapper.profile.centre_0])
        )

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 1.0
        assert mappers[0].profile.centre_0.upper_limit == 1.05

        assert mappers[-1].profile.centre_0.lower_limit == 1.45
        assert mappers[-1].profile.centre_0.upper_limit == 1.5

    def test_raises_exception_for_bad_limits(self, grid_search, mapper):
        mapper.profile.centre_0 = p.GaussianPrior(
            0.0, 2.0, lower_limit=float("-inf"), upper_limit=float("inf")
        )
        with pytest.raises(exc.PriorException):
            list(
                grid_search.make_arguments(
                    [[0, 1]], grid_priors=[mapper.profile.centre_0]
                )
            )


init_args = []
fit_args = []
fit_instances = []


class MockOptimizer(autofit.optimize.non_linear.non_linear.NonLinearOptimizer):
    def __init__(self, paths):
        super().__init__(paths)
        init_args.append(paths.phase_name)

    def fit(self, analysis, model):
        fit_args.append(analysis)
        # noinspection PyTypeChecker
        return autofit.optimize.non_linear.non_linear.Result(
            None, analysis.fit(None), None
        )


class MockAnalysis(autofit.optimize.non_linear.non_linear.Analysis):
    def fit(self, instance):
        fit_instances.append(instance)
        return 1

    def visualize(self, instance, during_analysis):
        pass

    def log(self, instance):
        pass


class MockClassContainer(object):
    def __init__(self):
        self.init_args = init_args
        self.fit_args = fit_args
        self.fit_instances = fit_instances

        self.MockOptimizer = MockOptimizer
        self.MockAnalysis = MockAnalysis


@pytest.fixture(name="container")
def make_mock_class_container():
    init_args.clear()
    fit_args.clear()
    fit_instances.clear()
    return MockClassContainer()


@pytest.fixture(name="grid_search_05")
def make_grid_search_05(container):
    return gs.GridSearch(
        optimizer_class=container.MockOptimizer,
        number_of_steps=2,
        paths=Paths(phase_name="sample_name"),
    )


class TestGridNLOBehaviour(object):
    def test_calls(self, grid_search_05, container, mapper):

        result = grid_search_05.fit(
            container.MockAnalysis(), mapper, [mapper.profile.centre_0]
        )

        assert len(container.init_args) == 2
        assert len(container.fit_args) == 2
        assert len(result.results) == 2

    def test_names_1d(self, grid_search_05, container, mapper):
        grid_search_05.fit(container.MockAnalysis(), mapper, [mapper.profile.centre_0])

        assert len(container.init_args) == 2
        assert container.init_args[0] == "sample_name//profile_centre_0_0.00_0.50"
        assert container.init_args[1] == "sample_name//profile_centre_0_0.50_1.00"

    def test_round_names(self, container, mapper):
        grid_search = gs.GridSearch(
            optimizer_class=container.MockOptimizer,
            number_of_steps=3,
            paths=Paths(phase_name="sample_name"),
        )

        grid_search.fit(container.MockAnalysis(), mapper, [mapper.profile.centre_0])

        assert len(container.init_args) == 3
        assert container.init_args[0] == "sample_name//profile_centre_0_0.00_0.33"
        assert container.init_args[1] == "sample_name//profile_centre_0_0.33_0.67"
        assert container.init_args[2] == "sample_name//profile_centre_0_0.67_1.00"

    def test_names_2d(self, grid_search_05, mapper, container):
        grid_search_05.fit(
            container.MockAnalysis(),
            mapper,
            [mapper.profile.centre_0, mapper.profile.centre_1],
        )

        assert len(container.init_args) == 4

        sorted_args = list(sorted(container.init_args[n] for n in range(4)))

        assert (
            sorted_args[0]
            == "sample_name//profile_centre_0_0.00_0.50_profile_centre_1_0.00_0.50"
        )
        assert (
            sorted_args[1]
            == "sample_name//profile_centre_0_0.00_0.50_profile_centre_1_0.50_1.00"
        )
        assert (
            sorted_args[2]
            == "sample_name//profile_centre_0_0.50_1.00_profile_centre_1_0.00_0.50"
        )
        assert (
            sorted_args[3]
            == "sample_name//profile_centre_0_0.50_1.00_profile_centre_1_0.50_1.00"
        )

    def test_results(self, grid_search_05, mapper, container):
        result = grid_search_05.fit(
            container.MockAnalysis(),
            mapper,
            [mapper.profile.centre_0, mapper.profile.centre_1],
        )

        assert len(result.results) == 4
        assert result.no_dimensions == 2
        assert np.equal(
            result.figure_of_merit_array, np.array([[1.0, 1.0], [1.0, 1.0]])
        ).all()

        grid_search = gs.GridSearch(
            optimizer_class=container.MockOptimizer,
            number_of_steps=10,
            paths=Paths(phase_name="sample_name"),
        )
        result = grid_search.fit(
            container.MockAnalysis(),
            mapper,
            [mapper.profile.centre_0, mapper.profile.centre_1],
        )

        assert len(result.results) == 100
        assert result.no_dimensions == 2
        assert result.figure_of_merit_array.shape == (10, 10)

    def test_results_parallel(self, mapper, container):
        grid_search = gs.GridSearch(
            optimizer_class=container.MockOptimizer,
            number_of_steps=10,
            paths=Paths(phase_name="sample_name"),
            parallel=True,
        )
        result = grid_search.fit(
            container.MockAnalysis(),
            mapper,
            [mapper.profile.centre_0, mapper.profile.centre_1],
        )

        assert len(result.results) == 100
        assert result.no_dimensions == 2
        assert result.figure_of_merit_array.shape == (10, 10)

    def test_generated_models_with_instances(self, grid_search, container, mapper):
        instance_profile = GeometryProfile()
        mapper.instance_profile = instance_profile

        analysis = container.MockAnalysis()

        grid_search.fit(analysis, mapper, [mapper.profile.centre_0])

        for instance in container.fit_instances:
            assert isinstance(instance.profile, GeometryProfile)
            assert instance.instance_profile == instance_profile

    def test_generated_models_with_instance_attributes(
        self, grid_search, mapper, container
    ):
        instance = 2.0
        mapper.profile.centre_1 = instance

        analysis = container.MockAnalysis()

        grid_search.fit(analysis, mapper, [mapper.profile.centre_0])

        assert len(container.fit_instances) > 0

        for instance in container.fit_instances:
            assert isinstance(instance.profile, GeometryProfile)
            # noinspection PyUnresolvedReferences
            assert instance.profile.centre[1] == 2

    def test_passes_attributes(self):
        grid_search = gs.GridSearch(
            Paths(phase_name=""),
            number_of_steps=10,
            optimizer_class=autofit.optimize.non_linear.multi_nest.MultiNest,
        )

        grid_search.n_live_points = 20
        grid_search.sampling_efficiency = 0.3

        optimizer = grid_search.optimizer_instance("name_path")

        assert optimizer.n_live_points is grid_search.n_live_points
        assert optimizer.sampling_efficiency is grid_search.sampling_efficiency
        assert grid_search.paths.path != optimizer.paths.path
        assert grid_search.paths.backup_path != optimizer.paths.backup_path
        assert grid_search.paths.phase_output_path != optimizer.paths.phase_output_path


class MockResult(object):
    def __init__(self, figure_of_merit):
        self.figure_of_merit = figure_of_merit
        self.model = figure_of_merit


@pytest.fixture(name="grid_search_result")
def make_grid_search_result():
    one = MockResult(1)
    two = MockResult(2)

    return gs.GridSearchResult([one, two], [[1], [2]])


class TestGridSearchResult(object):
    def test_best_result(self, grid_search_result):
        assert grid_search_result.best_result.figure_of_merit == 2

    def test_attributes(self, grid_search_result):
        assert grid_search_result.model == 2

    def test_best_model(self, grid_search_result):
        assert grid_search_result.best_model == 2

    def test_all_models(self, grid_search_result):
        assert grid_search_result.all_models == [1, 2]


class TestMixin(object):
    def test_mixin(self, container):
        class MyPhase(phase.as_grid_search(phase.AbstractPhase)):
            @property
            def grid_priors(self):
                return [self.model.profile.centre_0]

            def run(self):
                analysis = container.MockAnalysis()
                return self.make_result(self.run_analysis(analysis), analysis)

        my_phase = MyPhase(
            Paths(phase_name="", phase_folders=tuple()),
            number_of_steps=2,
            optimizer_class=container.MockOptimizer,
        )
        my_phase.model.profile = GeometryProfile

        result = my_phase.run()

        assert isinstance(result, gs.GridSearchResult)
        assert len(result.results) == 2

        assert isinstance(
            result.best_result, autofit.optimize.non_linear.non_linear.Result
        )

    def test_parallel_flag(self):
        my_phase = phase.as_grid_search(phase.AbstractPhase, parallel=True)(
            Paths(phase_name="phase name")
        )
        assert my_phase.optimizer.parallel
