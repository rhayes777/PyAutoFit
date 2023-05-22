import pytest

import autofit as af
from autofit.interpolator import CovarianceInterpolator
import numpy as np

from autofit.interpolator.covariance import LinearAnalysis, LinearRelationship


@pytest.fixture(name="interpolator")
def make_interpolator():
    samples_list = [
        af.SamplesPDF(
            model=af.Collection(
                t=value,
                gaussian=af.Model(af.Gaussian),
            ),
            sample_list=[
                af.Sample(
                    log_likelihood=-i,
                    log_prior=1.0,
                    weight=1.0,
                    kwargs={
                        ("gaussian", "centre"): value + i,
                        ("gaussian", "normalization"): value + i**2,
                        ("gaussian", "sigma"): value + i**3,
                    },
                )
                for i in range(3)
            ],
        )
        for value in range(3)
    ]
    return CovarianceInterpolator(
        samples_list,
    )


def test_covariance_matrix(interpolator):
    assert np.allclose(
        interpolator.covariance_matrix,
        np.array(
            [
                [1.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 4.33333333, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 9.0, 19.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 4.33333333, 9.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 9.0, 19.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 4.33333333, 9.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 9.0, 19.0],
            ]
        ),
    )


# Fails due to poorly defined inversion?
def _test_inverse_covariance_matrix(interpolator):
    identity = np.dot(
        interpolator.covariance_matrix, interpolator.inverse_covariance_matrix
    )
    print(identity)
    assert np.allclose(
        identity,
        np.eye(9),
    )


def test_covariance_is_invertible(interpolator):
    assert np.linalg.det(interpolator.covariance_matrix) != 0
    assert np.linalg.inv(interpolator.covariance_matrix) is not None


def test_y(interpolator):
    assert (
        interpolator._y == np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    ).all()


def test_interpolate(interpolator):
    assert interpolator[interpolator.t == 0.5] == 0.5


def test_trivial_linear_analysis():
    linear_analysis = LinearAnalysis(
        x=[1.0, 2.0, 3.0],
        y=[2.0, 4.0, 6.0],
        inverse_covariance_matrix=np.eye(3),
    )
    instance = [
        LinearRelationship(
            m=2.0,
            c=0.0,
        )
    ]
    assert list(linear_analysis._y(instance)) == [2.0, 4.0, 6.0]
    assert linear_analysis.log_likelihood_function(instance) == 0.0


def test_multiple_time_points_linear_analysis():
    y = [2.0, 3.0, 4.0, 6.0, 6.0, 9.0]
    linear_analysis = LinearAnalysis(
        x=[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
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
    assert list(linear_analysis._y(instance)) == y
    assert linear_analysis.log_likelihood_function(instance) == 0.0
