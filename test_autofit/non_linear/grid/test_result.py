import pytest

import autofit as af


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
        af.Gaussian,
        centre=af.UniformPrior(
            lower_limit=0.0,
            upper_limit=1.0
        )
    )


@pytest.fixture(
    name="result"
)
def make_result(model):
    return af.GridSearchResult(
        results=[],
        lower_limits_lists=[
            [0.0],
            [0.5]
        ],
        grid_priors=[model.centre]
    )


@pytest.mark.parametrize(
    "upper_limit, physical_value",
    [
        (1.0, 0.5),
        (2.0, 1.0),
        (4.0, 2.0),
    ]
)
def test_physical_lower_limits(
        upper_limit,
        physical_value,
        model,
        result
):
    result.grid_priors = [
        af.UniformPrior(
            lower_limit=0.0,
            upper_limit=upper_limit
        )
    ]
    assert result.physical_lower_limits_lists == [
        [0.0],
        [physical_value]
    ]


def test_limits_lists(result):
    assert result.lower_limits_lists == [
        [0.0], [0.5]
    ]
    assert result.upper_limits_lists == [
        [0.5], [1.0]
    ]


def test_physical_centres_lists(
        result
):
    assert result.physical_centres_lists == [
        [0.25], [0.75]
    ]


def test_physical_upper_limits_lists(
        result
):
    assert result.physical_upper_limits_lists == [
        [0.5], [1.0]
    ]


def test_log_uniform_prior(
        result
):
    result.grid_priors = [af.LogUniformPrior()]

    assert result.physical_lower_limits_lists == [[1e-06], [pytest.approx(0.001, rel=0.01)]]
    assert result.physical_centres_lists == [
        [pytest.approx(3.1622776601683795e-05, rel=0.01)], [pytest.approx(0.03162277660168379, rel=0.01)]
    ]
    assert result.physical_upper_limits_lists == [[pytest.approx(0.001, rel=0.01)], [pytest.approx(1.0, rel=0.01)]]
