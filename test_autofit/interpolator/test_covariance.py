from autofit.interpolator import CovarianceInterpolator
import numpy as np


def test_covariance_matrix(instances):
    interpolator = CovarianceInterpolator(
        instances=instances,
        covariance_matrices=[np.array([[1.0, 0.0], [0.0, 1.0]])],
    )
