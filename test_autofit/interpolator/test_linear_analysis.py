import numpy as np
import pytest

from autofit.interpolator.covariance import LinearAnalysis, LinearRelationship


@pytest.mark.parametrize(
    "m, c, y",
    [
        (2.0, 0.0, [2.0, 4.0, 6.0]),
        (1.0, 0.0, [1.0, 2.0, 3.0]),
        (1.0, 1.0, [2.0, 3.0, 4.0]),
    ],
)
def test_trivial_linear_analysis(m, c, y):
    linear_analysis = LinearAnalysis(
        x=[1.0, 2.0, 3.0],
        y=y,
        inverse_covariance_matrix=np.eye(3),
    )
    instance = [
        LinearRelationship(
            m=m,
            c=c,
        )
    ]
    assert list(linear_analysis._y(instance)) == y
    assert linear_analysis.log_likelihood_function(instance) == 0.0


@pytest.fixture(
    name="two_times",
)
def make_two_times():
    return LinearRelationship(
        m=2.0,
        c=0.0,
    )


@pytest.mark.parametrize(
    "x, y",
    [
        ([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [2.0, 3.0, 4.0, 6.0, 6.0, 9.0]),
        ([2.0, 2.0, 1.0, 1.0, 3.0, 3.0], [4.0, 6.0, 2.0, 3.0, 6.0, 9.0]),
    ],
)
def test_multiple_time_points_linear_analysis(x, y, two_times):
    linear_analysis = LinearAnalysis(
        x=x,
        y=y,
        inverse_covariance_matrix=np.eye(6),
    )
    instance = [
        two_times,
        LinearRelationship(
            m=3.0,
            c=0.0,
        ),
    ]
    assert list(linear_analysis._y(instance)) == [2.0, 3.0, 4.0, 6.0, 6.0, 9.0]
    assert linear_analysis.log_likelihood_function(instance) == 0.0


def test_non_trivial_covariance(two_times):
    linear_analysis = LinearAnalysis(
        x=[1.0, 2.0],
        y=[2.5, 3.0],
        inverse_covariance_matrix=np.array(
            [
                [2.0, 1.0],
                [1.0, 2.0],
            ]
        ),
    )
    instance = [two_times]
    assert linear_analysis.log_likelihood_function(instance) == -0.75
