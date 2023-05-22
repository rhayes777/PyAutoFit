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


@pytest.mark.parametrize(
    "x, y",
    [
        ([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], [2.0, 3.0, 4.0, 6.0, 6.0, 9.0]),
        ([2.0, 2.0, 1.0, 1.0, 3.0, 3.0], [4.0, 6.0, 2.0, 3.0, 6.0, 9.0]),
    ],
)
def test_multiple_time_points_linear_analysis(x, y):
    linear_analysis = LinearAnalysis(
        x=x,
        y=y,
        inverse_covariance_matrix=np.eye(6),
    )
    instance = [
        LinearRelationship(
            m=2.0,
            c=0.0,
        ),
        LinearRelationship(
            m=3.0,
            c=0.0,
        ),
    ]
    assert list(linear_analysis._y(instance)) == [2.0, 3.0, 4.0, 6.0, 6.0, 9.0]
    assert linear_analysis.log_likelihood_function(instance) == 0.0
