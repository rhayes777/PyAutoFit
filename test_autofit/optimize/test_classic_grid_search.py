import os
import shutil

import pytest

import autofit.optimize.non_linear.grid_search
import autofit.optimize.non_linear.non_linear
from autofit import conf, Paths
from autofit import exc
from autofit.optimize.optimizer import grid
from test_autofit.mock import Galaxy


@pytest.fixture(scope="session", autouse=True)
def do_something():
    conf.instance = conf.Config(
        "{}/../../workspace/config".format(os.path.dirname(os.path.realpath(__file__)))
    )


class MockAnalysis(autofit.optimize.non_linear.non_linear.Analysis):
    def __init__(self):
        self.instances = []

    def fit(self, instance):
        self.instances.append(instance)
        try:
            return -(
                (instance.one.redshift - 0.1) ** 2 + (instance.two.redshift - 0.7) ** 2
            )
        except AttributeError:
            return 0

    def visualize(self, instance, during_analysis):
        pass


def tuple_lists_equal(l1, l2):
    assert len(l1) == len(l2)
    for tuple_pair in zip(l1, l2):
        assert len(tuple_pair[0]) == len(tuple_pair[1])
        for item in zip(tuple_pair[0], tuple_pair[1]):
            if item[0] != pytest.approx(item[1]):
                return False
    return True


class TestGridSearchOptimizer(object):
    def test_config(self):
        assert (
            autofit.optimize.non_linear.grid_search.GridSearch(
                Paths(phase_name="")
            ).step_size
            == 0.1
        )

    def test_1d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 1, 0.1)

        assert 10 == len(points)
        assert tuple_lists_equal(
            [
                (0.05,),
                (0.15,),
                (0.25,),
                (0.35,),
                (0.45,),
                (0.55,),
                (0.65,),
                (0.75,),
                (0.85,),
                (0.95,),
            ],
            points,
        )

    def test_2d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 2, 0.3)

        assert 9 == len(points)
        assert tuple_lists_equal(
            [
                (0.15, 0.15),
                (0.15, 0.45),
                (0.15, 0.75),
                (0.45, 0.15),
                (0.45, 0.45),
                (0.45, 0.75),
                (0.75, 0.15),
                (0.75, 0.45),
                (0.75, 0.75),
            ],
            points,
        )

    def test_3d(self):
        points = []

        def fit(point):
            points.append(point)
            return 0

        grid(fit, 3, 0.5)

        assert 3 == len(points[0])
        assert 8 == len(points)

    def test_best_fit(self):
        best_point = (0.65, 0.35)

        def fit(point):
            return -((best_point[0] - point[0]) ** 2 + (best_point[1] - point[1]) ** 2)

        result = grid(fit, 2, 0.1)

        assert pytest.approx(result[0]) == best_point[0]
        assert pytest.approx(result[1]) == best_point[1]


@pytest.fixture(name="grid_search")
def make_grid_search():
    name = "grid_search"
    try:
        shutil.rmtree("{}/{}/".format(conf.instance.output_path, name))
    except FileNotFoundError:
        pass
    return autofit.optimize.non_linear.grid_search.GridSearch(
        Paths(phase_name=name, remove_files=False), step_size=0.1
    )


class TestGridSearch(object):
    def test_1d(self, grid_search, model):
        model.one = Galaxy

        analysis = MockAnalysis()
        grid_search.fit(analysis, model)

        assert len(analysis.instances) == 10

        instance = analysis.instances[5]

        assert isinstance(instance.one, Galaxy)
        assert instance.one.redshift == 0.55

    def test_2d(self, grid_search, model):
        model.one = Galaxy
        model.two = Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis, model)

        assert pytest.approx(result.instance.one.redshift) == 0.05
        assert pytest.approx(result.instance.two.redshift) == 0.65

    def test_checkpoint_properties(self, grid_search, model):
        analysis = MockAnalysis()

        model.one = Galaxy
        grid_search.fit(analysis, model)

        paths = Paths(phase_name="grid_search")

        grid_search = autofit.optimize.non_linear.grid_search.GridSearch(
            paths, step_size=0.1
        )

        assert grid_search.is_checkpoint
        assert grid_search.checkpoint_count == 10
        assert grid_search.checkpoint_fit == 0.0
        assert grid_search.checkpoint_cube == (0.05,)
        assert grid_search.checkpoint_step_size == 0.1
        assert grid_search.checkpoint_prior_count == 1

    def test_recover_bad_checkpoint(self, grid_search, model):
        analysis = MockAnalysis()

        model.one = Galaxy
        grid_search.fit(analysis, model)

        grid_search = autofit.optimize.non_linear.grid_search.GridSearch(
            Paths(phase_name="grid_search"), step_size=0.1
        )

        model.two = Galaxy

        with pytest.raises(exc.CheckpointException):
            grid_search.fit(analysis, model)

        grid_search = autofit.optimize.non_linear.grid_search.GridSearch(
            Paths(phase_name="grid_search"), step_size=0.2
        )
        model.one = Galaxy

        with pytest.raises(exc.CheckpointException):
            grid_search.fit(analysis, model)

    def test_recover_checkpoint(self, grid_search, model):
        analysis = MockAnalysis()

        model.one = Galaxy
        model.two = Galaxy

        grid_search.fit(analysis, model)

        grid_search = autofit.optimize.non_linear.grid_search.GridSearch(
            Paths(phase_name="grid_search"), step_size=0.1
        )

        analysis = MockAnalysis()

        result = grid_search.fit(analysis, model)

        assert len(analysis.instances) == 1
        assert pytest.approx(result.instance.one.redshift) == 0.05
        assert pytest.approx(result.instance.two.redshift) == 0.65

    def test_recover_midway(self, grid_search, model):
        string = "11\n0\n(0.0, 0.0)\n0.1\n2"
        with open(grid_search.checkpoint_path, "w+") as f:
            f.write(string)

        grid_search = autofit.optimize.non_linear.grid_search.GridSearch(
            Paths(phase_name="grid_search"), step_size=0.1
        )

        model.one = Galaxy
        model.two = Galaxy

        analysis = MockAnalysis()

        result = grid_search.fit(analysis, model)

        assert len(analysis.instances) == 90
        assert pytest.approx(result.instance.one.redshift) == 0.15
        assert pytest.approx(result.instance.two.redshift) == 0.65

    def test_instances(self, grid_search, model):
        model.one = Galaxy

        analysis = MockAnalysis()
        result = grid_search.fit(analysis, model)

        assert len(result.instances) == 10

        instance = result.instances[5]

        assert isinstance(instance[0].one, Galaxy)
        assert instance[0].one.redshift == 0.55
