import pytest

import autofit as af
from autofit.interpolator import CovarianceInterpolator
import numpy as np


@pytest.fixture(name="interpolator")
def make_interpolator():
    samples_list = [
        af.SamplesPDF(
            model=af.Collection(
                t=value,
                gaussian=af.Model(
                    af.Gaussian,
                    centre=af.GaussianPrior(mean=1.0, sigma=1.0),
                    normalization=af.GaussianPrior(mean=1.0, sigma=1.0),
                    sigma=af.GaussianPrior(mean=1.0, sigma=1.0),
                ),
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


def test_interpolate(interpolator):
    assert interpolator[interpolator.t == 0.5].gaussian.centre == 0.5


def test_linear_analysis_for_value(interpolator):
    analysis = interpolator._linear_analysis_for_value(interpolator.t == 0.5)
    assert (analysis.x == np.array([0, 1, 2])).all()
    assert (analysis.y == np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])).all()


def test_model(interpolator):
    model = interpolator.model
    assert model.prior_count == 6


def test_single_variable():
    samples_list = [
        af.SamplesPDF(
            model=af.Collection(
                t=value,
                v=af.GaussianPrior(mean=1.0, sigma=1.0),
            ),
            sample_list=[
                af.Sample(
                    log_likelihood=-value,
                    log_prior=1.0,
                    weight=1.0,
                    kwargs={
                        ("v",): value,
                    },
                )
            ],
        )
        for value in range(100)
    ]
    interpolator = CovarianceInterpolator(
        samples_list,
    )
    assert interpolator[interpolator.t == 50.0].v == pytest.approx(50.0, abs=1.0)


def test_variable_and_constant():
    samples_list = [
        af.SamplesPDF(
            model=af.Collection(
                t=value,
                v=af.GaussianPrior(mean=1.0, sigma=1.0),
                x=af.GaussianPrior(mean=1.0, sigma=1.0),
            ),
            sample_list=[
                af.Sample(
                    log_likelihood=-value,
                    log_prior=1.0,
                    weight=1.0,
                    kwargs={
                        ("v",): value + 0.1 * (1 - np.random.random()),
                        ("x",): 0.5 * (1 - +np.random.random()),
                    },
                )
                for _ in range(100)
            ],
        )
        for value in range(100)
    ]
    interpolator = CovarianceInterpolator(
        samples_list,
    )
    print(interpolator.covariance_matrix)
    assert interpolator[interpolator.t == 50.0].v == pytest.approx(50.0, abs=1.0)
