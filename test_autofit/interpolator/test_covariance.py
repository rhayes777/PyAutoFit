import autofit as af
from autofit.interpolator import CovarianceInterpolator
import numpy as np


def test_covariance_matrix(instances):
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
                        ("gaussian", "normalization"): value + i,
                        ("gaussian", "sigma"): value + i,
                    },
                )
                for i in range(3)
            ],
        )
        for value in range(3)
    ]
    interpolator = CovarianceInterpolator(
        samples_list,
    )
    assert (
        interpolator.covariance_matrix
        == np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )
    ).all()
