import pytest

import autofit as af
from autofit.non_linear.samples.summary import SamplesSummary


@pytest.fixture(name="model")
def make_model():
    return af.Model(
        af.Gaussian, centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    )


@pytest.fixture(name="samples_1")
def make_samples_1(model):
    return SamplesSummary(
        af.Sample(
            1.0,
            1.0,
            1.0,
            {
                "centre": 1.0,
                "normalization": 2.0,
                "sigma": 3.0,
            },
        ),
        model,
    )


@pytest.fixture(name="samples_2")
def make_samples_2(model):
    return SamplesSummary(
        af.Sample(
            1.0,
            1.0,
            1.0,
            {
                "centre": 2.0,
                "normalization": 4.0,
                "sigma": 6.0,
            },
        ),
        model,
    )


@pytest.fixture(name="result")
def make_result(model, samples_1, samples_2):
    return af.GridSearchResult(
        samples=[samples_1, samples_2],
        lower_limits_lists=[[0.0], [0.5]],
        grid_priors=[model.centre],
    )


def test_instance_attribute_from_path(result):
    assert (result.attribute_grid("centre").native == [1.0, 2.0]).all()


def test_higher_dimension_instance_attributes(model, samples_1, samples_2):
    result = af.GridSearchResult(
        samples=[samples_1, samples_2, samples_1, samples_2],
        lower_limits_lists=[[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]],
        grid_priors=[model.centre, model.normalization],
    )
    assert (result.attribute_grid("centre").native == [[1.0, 2.0], [1.0, 2.0]]).all()


@pytest.fixture(name="deep_result")
def make_deep_result(model, samples_1, samples_2):
    model = af.Collection(
        gaussian=af.Model(
            af.Gaussian,
            centre=af.UniformPrior(lower_limit=0.0, upper_limit=1.0),
        )
    )
    samples = SamplesSummary(
        af.Sample(
            1.0,
            1.0,
            1.0,
            {
                "gaussian.centre": 1.0,
                "gaussian.normalization": 2.0,
                "gaussian.sigma": 3.0,
            },
        ),
        model,
    )
    return af.GridSearchResult(
        samples=[samples, samples],
        lower_limits_lists=[[0.0], [0.5]],
        grid_priors=[model.gaussian.centre],
    )


def test_paths(deep_result):
    assert (deep_result.attribute_grid("gaussian.centre").native == [1.0, 1.0]).all()


def test_instances(deep_result):
    assert (
        deep_result.attribute_grid("gaussian").native
        == [af.Gaussian(1.0, 2.0, 3.0), af.Gaussian(1.0, 2.0, 3.0)]
    ).all()


@pytest.mark.parametrize(
    "upper_limit, physical_value",
    [
        (1.0, 0.5),
        (2.0, 1.0),
        (4.0, 2.0),
    ],
)
def test_physical_lower_limits(upper_limit, physical_value, model, result):
    result.grid_priors = [af.UniformPrior(lower_limit=0.0, upper_limit=upper_limit)]
    assert result.physical_lower_limits_lists == [[0.0], [physical_value]]


def test_limits_lists(result):
    assert result.lower_limits_lists == [[0.0], [0.5]]
    assert result.upper_limits_lists == [[0.5], [1.0]]


def test_physical_centres_lists(result):
    assert result.physical_centres_lists == [[0.25], [0.75]]


def test_physical_upper_limits_lists(result):
    assert result.physical_upper_limits_lists == [[0.5], [1.0]]


def test_log_uniform_prior(result):
    result.grid_priors = [af.LogUniformPrior()]

    assert result.physical_lower_limits_lists == [
        [1e-06],
        [pytest.approx(0.001, rel=0.01)],
    ]
    assert result.physical_centres_lists == [
        [pytest.approx(3.1622776601683795e-05, rel=0.01)],
        [pytest.approx(0.03162277660168379, rel=0.01)],
    ]
    assert result.physical_upper_limits_lists == [
        [pytest.approx(0.001, rel=0.01)],
        [pytest.approx(1.0, rel=0.01)],
    ]
