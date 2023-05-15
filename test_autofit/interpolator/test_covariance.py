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
                    log_likelihood=1.0,
                    log_prior=1.0,
                    weight=1.0,
                    kwargs={
                        ("gaussian", "centre"): value,
                        ("gaussian", "normalization"): value,
                        ("gaussian", "sigma"): value,
                    },
                )
                for _ in range(3)
            ],
        )
        for value in range(3)
    ]
    interpolator = CovarianceInterpolator(
        samples_list,
    )
    assert interpolator.covariance_matrix.shape == (9, 9)
