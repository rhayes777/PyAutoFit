import os
import pytest

import autofit as af


class MockFitness:

    def __init__(self, figure_of_merit=0.0, increase_figure_of_merit = True):

        self.figure_of_merit = figure_of_merit
        self.increase_figure_of_merit = increase_figure_of_merit

    def __call__(self, parameters):

        if self.increase_figure_of_merit:
            self.figure_of_merit += 1

        return self.figure_of_merit


def test__priors__samples_from_model():
    model = af.Model(af.m.MockClassx4)
    model.one = af.UniformPrior(lower_limit=0.099, upper_limit=0.101)
    model.two = af.UniformPrior(lower_limit=0.199, upper_limit=0.201)
    model.three = af.UniformPrior(lower_limit=0.299, upper_limit=0.301)
    model.four = af.UniformPrior(lower_limit=0.399, upper_limit=0.401)

    initializer = af.InitializerPrior()

    unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_from_model(
        total_points=2, model=model, fitness=MockFitness()
    )

    assert 0.0 < unit_parameter_lists[0][0] < 1.0
    assert 0.0 < unit_parameter_lists[1][0] < 1.0
    assert 0.0 < unit_parameter_lists[0][1] < 1.0
    assert 0.0 < unit_parameter_lists[1][1] < 1.0
    assert 0.0 < unit_parameter_lists[0][2] < 1.0
    assert 0.0 < unit_parameter_lists[1][2] < 1.0
    assert 0.0 < unit_parameter_lists[0][3] < 1.0
    assert 0.0 < unit_parameter_lists[1][3] < 1.0

    assert 0.099 < parameter_lists[0][0] < 0.101
    assert 0.099 < parameter_lists[1][0] < 0.101
    assert 0.199 < parameter_lists[0][1] < 0.201
    assert 0.199 < parameter_lists[1][1] < 0.201
    assert 0.299 < parameter_lists[0][2] < 0.301
    assert 0.299 < parameter_lists[1][2] < 0.301
    assert 0.399 < parameter_lists[0][3] < 0.401
    assert 0.399 < parameter_lists[1][3] < 0.401

    assert figure_of_merit_list == [1.0, 2.0]

def test__priors__samples_from_model__raise_exception_if_all_likelihoods_identical():
    model = af.Model(af.m.MockClassx4)

    initializer = af.InitializerPrior()

    with pytest.raises(af.exc.InitializerException):

        initializer.samples_from_model(
            total_points=2, model=model, fitness=MockFitness(increase_figure_of_merit=False)
        )

def test__priors__samples_in_test_mode():

    os.environ["PYAUTOFIT_TEST_MODE"] = "1"

    model = af.Model(af.m.MockClassx4)
    model.one = af.UniformPrior(lower_limit=0.099, upper_limit=0.101)
    model.two = af.UniformPrior(lower_limit=0.199, upper_limit=0.201)
    model.three = af.UniformPrior(lower_limit=0.299, upper_limit=0.301)
    model.four = af.UniformPrior(lower_limit=0.399, upper_limit=0.401)

    initializer = af.InitializerPrior()

    unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_from_model(
        total_points=2, model=model, fitness=None
    )

    assert 0.0 < unit_parameter_lists[0][0] < 1.0
    assert 0.0 < unit_parameter_lists[1][0] < 1.0
    assert 0.0 < unit_parameter_lists[0][1] < 1.0
    assert 0.0 < unit_parameter_lists[1][1] < 1.0
    assert 0.0 < unit_parameter_lists[0][2] < 1.0
    assert 0.0 < unit_parameter_lists[1][2] < 1.0
    assert 0.0 < unit_parameter_lists[0][3] < 1.0
    assert 0.0 < unit_parameter_lists[1][3] < 1.0

    assert 0.099 < parameter_lists[0][0] < 0.101
    assert 0.099 < parameter_lists[1][0] < 0.101
    assert 0.199 < parameter_lists[0][1] < 0.201
    assert 0.199 < parameter_lists[1][1] < 0.201
    assert 0.299 < parameter_lists[0][2] < 0.301
    assert 0.299 < parameter_lists[1][2] < 0.301
    assert 0.399 < parameter_lists[0][3] < 0.401
    assert 0.399 < parameter_lists[1][3] < 0.401

    assert figure_of_merit_list == [-1.0e99, -1.0e100]

    os.environ["PYAUTOFIT_TEST_MODE"] = "0"

def test__ball__samples_sample_centre_of_priors():

    model = af.Model(af.m.MockClassx4)
    model.one = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    model.two = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
    model.three = af.UniformPrior(lower_limit=0.0, upper_limit=3.0)
    model.four = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)

    initializer = af.InitializerBall(lower_limit=0.4999, upper_limit=0.5001)

    unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_from_model(
        total_points=2, model=model, fitness=MockFitness()
    )

    assert 0.4999 < unit_parameter_lists[0][0] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][0] < 0.5001
    assert 0.4999 < unit_parameter_lists[0][1] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][1] < 0.5001
    assert 0.4999 < unit_parameter_lists[0][2] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][2] < 0.5001
    assert 0.4999 < unit_parameter_lists[0][3] < 0.5001
    assert 0.4999 < unit_parameter_lists[1][3] < 0.5001

    assert 0.499 < parameter_lists[0][0] < 0.501
    assert 0.499 < parameter_lists[1][0] < 0.501
    assert 0.999 < parameter_lists[0][1] < 1.001
    assert 0.999 < parameter_lists[1][1] < 1.001
    assert 1.499 < parameter_lists[0][2] < 1.501
    assert 1.499 < parameter_lists[1][2] < 1.501
    assert 1.999 < parameter_lists[0][3] < 2.001
    assert 1.999 < parameter_lists[1][3] < 2.001

    initializer = af.InitializerBall(lower_limit=0.7999, upper_limit=0.8001)

    unit_parameter_lists, parameter_lists, figure_of_merit_list = initializer.samples_from_model(
        total_points=2, model=model, fitness=MockFitness()
    )

    assert 0.799 < parameter_lists[0][0] < 0.801
    assert 0.799 < parameter_lists[1][0] < 0.801
    assert 1.599 < parameter_lists[0][1] < 1.601
    assert 1.599 < parameter_lists[1][1] < 1.601
    assert 2.399 < parameter_lists[0][2] < 2.401
    assert 2.399 < parameter_lists[1][2] < 2.401
    assert 3.199 < parameter_lists[0][3] < 3.201
    assert 3.199 < parameter_lists[1][3] < 3.201

    assert figure_of_merit_list == [1.0, 2.0]


@pytest.mark.parametrize(
    "unit_value, physical_value",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
    ]
)
def test_invert_physical(unit_value, physical_value):
    prior = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=1.0,
    )
    assert prior.unit_value_for(unit_value) == pytest.approx(physical_value)


@pytest.mark.parametrize(
    "unit_value, physical_value",
    [
        (1.0, 0.0),
        (2.0, 0.5),
        (3.0, 1.0),
    ]
)
def test_invert_physical_offset(unit_value, physical_value):
    prior = af.UniformPrior(
        lower_limit=1.0,
        upper_limit=3.0,
    )
    assert prior.unit_value_for(unit_value) == pytest.approx(physical_value)


@pytest.mark.parametrize(
    "unit_value, physical_value",
    [
        (-float("inf"), 0.0),
        (0.0, 0.5),
        (float("inf"), 1.0),
    ]
)
def test_invert_gaussian(unit_value, physical_value):
    prior = af.GaussianPrior(
        mean=0.0,
        sigma=3.0,
    )
    assert prior.unit_value_for(unit_value) == physical_value


@pytest.fixture(name="model")
def make_model():
    return af.Model(
        af.Gaussian,
        centre=af.UniformPrior(1.0, 2.0),
        normalization=af.UniformPrior(2.0, 3.0),
        sigma=af.UniformPrior(-2.0, -1.0),
    )


def test_starting_point_initializer(model):
    initializer = af.SpecificRangeInitializer({
        model.centre: (1.0, 2.0),
        model.normalization: (2.0, 3.0),
        model.sigma: (-2.0, -1.0),
    })

    parameter_list = initializer._generate_unit_parameter_list(model)
    assert len(parameter_list) == 3
    for parameter in parameter_list:
        assert 0.0 <= parameter <= 1.0


def test_offset(model):
    initializer = af.SpecificRangeInitializer({
        model.centre: (1.5, 2.0),
        model.normalization: (2.5, 3.0),
        model.sigma: (-1.5, -1.0),
    })

    parameter_list = initializer._generate_unit_parameter_list(model)
    assert len(parameter_list) == 3
    for parameter in parameter_list:
        assert 0.5 <= parameter <= 1.0


def test_missing_parameter(model):
    initializer = af.SpecificRangeInitializer(
        {
            model.centre: (1.5, 2.0),
            model.normalization: (2.5, 3.0),
        },
        lower_limit=0.5,
        upper_limit=0.5,
    )
    parameter_list = initializer._generate_unit_parameter_list(model)

    assert len(parameter_list) == 3
    for parameter in parameter_list:
        assert 0.5 <= parameter <= 1.0

    assert 0.5 in parameter_list
