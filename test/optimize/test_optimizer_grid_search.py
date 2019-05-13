import numpy as np
import pytest

from autofit import exc
from autofit import mock
from autofit.mapper import model_mapper as mm
from autofit.mapper import prior as p
from autofit.optimize import grid_search as gs
from autofit.optimize import non_linear
from autofit.tools import phase


@pytest.fixture(name="mapper")
def make_mapper():
    mapper = mm.ModelMapper()
    mapper.profile = mock.GeometryProfile
    return mapper


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    return gs.GridSearch(phase_name='', model_mapper=mapper, number_of_steps=10)


class TestGridSearchablePriors(object):

    def test_generated_models(self, grid_search):
        mappers = list(grid_search.model_mappers(
            grid_priors=[grid_search.variable.profile.centre_0, grid_search.variable.profile.centre_1]))

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
        grid_search = gs.GridSearch(phase_name='', model_mapper=mapper, number_of_steps=10)

        mappers = list(grid_search.model_mappers(grid_priors=[mapper.profile.centre_0]))

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 1.0

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.0
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

    def test_tied_priors(self, grid_search):
        grid_search.variable.profile.centre_0 = grid_search.variable.profile.centre_1

        mappers = list(grid_search.model_mappers(
            grid_priors=[grid_search.variable.profile.centre_0, grid_search.variable.profile.centre_1]))

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

    def test_different_prior_width(self, grid_search):
        grid_search.variable.profile.centre_0 = p.UniformPrior(0., 2.)
        mappers = list(grid_search.model_mappers(
            grid_priors=[grid_search.variable.profile.centre_0]))

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.2

        assert mappers[-1].profile.centre_0.lower_limit == 1.8
        assert mappers[-1].profile.centre_0.upper_limit == 2.0

        grid_search.variable.profile.centre_0 = p.UniformPrior(1., 1.5)
        mappers = list(grid_search.model_mappers(
            grid_priors=[grid_search.variable.profile.centre_0]))

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 1.0
        assert mappers[0].profile.centre_0.upper_limit == 1.05

        assert mappers[-1].profile.centre_0.lower_limit == 1.45
        assert mappers[-1].profile.centre_0.upper_limit == 1.5

    def test_raises_exception_for_bad_limits(self, grid_search):
        grid_search.variable.profile.centre_0 = p.GaussianPrior(0., 2., lower_limit=float('-inf'),
                                                                upper_limit=float('inf'))
        with pytest.raises(exc.PriorException):
            list(grid_search.make_arguments([[0, 1]], grid_priors=[grid_search.variable.profile.centre_0]))


init_args = []
fit_args = []
fit_instances = []


class MockOptimizer(non_linear.NonLinearOptimizer):
    def __init__(self, phase_name="mock_optimizer", phase_tag="tag", phase_folders=None, model_mapper=None):
        super().__init__(phase_folders=phase_folders, phase_tag=phase_tag, phase_name=phase_name,
                         model_mapper=model_mapper)
        init_args.append((model_mapper, phase_name))

    def fit(self, analysis):
        fit_args.append(analysis)
        # noinspection PyTypeChecker
        return non_linear.Result(None, analysis.fit(None), None)


class MockAnalysis(non_linear.Analysis):
    def fit(self, instance):
        fit_instances.append(instance)
        return 1

    def visualize(self, instance, image_path, during_analysis):
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
def make_grid_search_05(mapper, container):
    return gs.GridSearch(model_mapper=mapper, optimizer_class=container.MockOptimizer, number_of_steps=2,
                         phase_name="sample_name")


class TestGridNLOBehaviour(object):

    def test_calls(self, grid_search_05, container, mapper):

        result = grid_search_05.fit(container.MockAnalysis(), [mapper.profile.centre_0])

        assert len(container.init_args) == 2
        assert len(container.fit_args) == 2
        assert len(result.results) == 2

    def test_names_1d(self, grid_search_05, container, mapper):
        grid_search_05.fit(container.MockAnalysis(), [mapper.profile.centre_0])

        assert len(container.init_args) == 2
        assert container.init_args[0][1] == "sample_name//profile_centre_0_0.00_0.50"
        assert container.init_args[1][1] == "sample_name//profile_centre_0_0.50_1.00"

    def test_round_names(self, container, mapper):
        grid_search = gs.GridSearch(model_mapper=mapper, optimizer_class=container.MockOptimizer, number_of_steps=3,
                                    phase_name="sample_name")

        grid_search.fit(container.MockAnalysis(), [mapper.profile.centre_0])

        assert len(container.init_args) == 3
        assert container.init_args[0][1] == "sample_name//profile_centre_0_0.00_0.33"
        assert container.init_args[1][1] == "sample_name//profile_centre_0_0.33_0.67"
        assert container.init_args[2][1] == "sample_name//profile_centre_0_0.67_1.00"

    def test_names_2d(self, grid_search_05, mapper, container):
        grid_search_05.fit(container.MockAnalysis(), [mapper.profile.centre_0, mapper.profile.centre_1])

        assert len(container.init_args) == 4

        sorted_args = list(sorted(container.init_args[n][1] for n in range(4)))

        assert sorted_args[0] == "sample_name//profile_centre_0_0.00_0.50_profile_centre_1_0.00_0.50"
        assert sorted_args[1] == "sample_name//profile_centre_0_0.00_0.50_profile_centre_1_0.50_1.00"
        assert sorted_args[2] == "sample_name//profile_centre_0_0.50_1.00_profile_centre_1_0.00_0.50"
        assert sorted_args[3] == "sample_name//profile_centre_0_0.50_1.00_profile_centre_1_0.50_1.00"

    def test_results(self, grid_search_05, mapper, container):
        result = grid_search_05.fit(container.MockAnalysis(), [mapper.profile.centre_0, mapper.profile.centre_1])

        assert len(result.results) == 4
        assert result.no_dimensions == 2
        assert np.equal(result.figure_of_merit_array, np.array([[1.0, 1.0],
                                                                [1.0, 1.0]])).all()

        grid_search = gs.GridSearch(model_mapper=mapper, optimizer_class=container.MockOptimizer, number_of_steps=10,
                                    phase_name="sample_name")
        result = grid_search.fit(container.MockAnalysis(), [mapper.profile.centre_0, mapper.profile.centre_1])

        assert len(result.results) == 100
        assert result.no_dimensions == 2
        assert result.figure_of_merit_array.shape == (10, 10)

    def test_results_parallel(self, mapper, container):
        grid_search = gs.GridSearch(model_mapper=mapper, optimizer_class=container.MockOptimizer, number_of_steps=10,
                                    phase_name="sample_name", parallel=True)
        result = grid_search.fit(container.MockAnalysis(), [mapper.profile.centre_0, mapper.profile.centre_1])

        assert len(result.results) == 100
        assert result.no_dimensions == 2
        assert result.figure_of_merit_array.shape == (10, 10)

    def test_generated_models_with_constants(self, grid_search, container):
        constant_profile = mock.GeometryProfile()
        grid_search.variable.constant_profile = constant_profile

        analysis = container.MockAnalysis()

        grid_search.fit(analysis, [grid_search.variable.profile.centre_0])

        for instance in container.fit_instances:
            assert isinstance(instance.profile, mock.GeometryProfile)
            assert instance.constant_profile == constant_profile

    def test_generated_models_with_constant_attributes(self, grid_search, container):
        constant = p.Constant(2)
        grid_search.variable.profile.centre_1 = constant

        analysis = container.MockAnalysis()

        grid_search.fit(analysis, [grid_search.variable.profile.centre_0])

        assert len(container.fit_instances) > 0

        for instance in container.fit_instances:
            assert isinstance(instance.profile, mock.GeometryProfile)
            # noinspection PyUnresolvedReferences
            assert instance.profile.centre[1] == 2

    def test_passes_attributes(self):
        grid_search = gs.GridSearch(phase_name='', model_mapper=mm.ModelMapper(), number_of_steps=10,
                                    optimizer_class=non_linear.MultiNest)

        grid_search.n_live_points = 20
        grid_search.sampling_efficiency = 0.3

        model_mapper = mm.ModelMapper()

        optimizer = grid_search.optimizer_instance(model_mapper, "name_path")

        assert optimizer.n_live_points is grid_search.n_live_points
        assert optimizer.sampling_efficiency is grid_search.sampling_efficiency
        assert optimizer.variable is model_mapper
        assert grid_search.path != optimizer.path
        assert grid_search.backup_path != optimizer.backup_path
        assert grid_search.phase_output_path != optimizer.phase_output_path


class MockResult(object):
    def __init__(self, figure_of_merit):
        self.figure_of_merit = figure_of_merit
        self.variable = figure_of_merit


@pytest.fixture(name="grid_search_result")
def make_grid_search_result():
    one = MockResult(1)
    two = MockResult(2)

    return gs.GridSearchResult([one, two], [[1], [2]])


class TestGridSearchResult(object):
    def test_best_result(self, grid_search_result):
        assert grid_search_result.best_result.figure_of_merit == 2

    def test_best_model(self, grid_search_result):
        assert grid_search_result.best_model == 2

    def test_all_models(self, grid_search_result):
        assert grid_search_result.all_models == [1, 2]


class TestMixin(object):
    def test_mixin(self, container):
        class MyPhase(phase.as_grid_search(phase.AbstractPhase)):
            @property
            def grid_priors(self):
                return [self.variable.profile.centre_0]

            def run(self):
                analysis = container.MockAnalysis()
                return self.make_result(self.run_analysis(analysis), analysis)

        optimizer = MyPhase(phase_name='', phase_folders=None, number_of_steps=2,
                            optimizer_class=container.MockOptimizer)
        optimizer.variable.profile = mock.GeometryProfile

        result = optimizer.run()

        assert isinstance(result, gs.GridSearchResult)
        assert len(result.results) == 2

        assert isinstance(result.best_result, non_linear.Result)

    def test_parallel_flag(self):
        my_phase = phase.as_grid_search(phase.AbstractPhase, parallel=True)(phase_name="phase name")
        assert my_phase.optimizer.parallel
