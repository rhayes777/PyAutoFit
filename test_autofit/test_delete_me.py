import numpy as np

import autofit as af
from autofit import DiagonalMatrix


def test_from_mode():
    message = af.UniformPrior(lower_limit=10, upper_limit=20).message
    mean = message.from_mode(14.03, covariance=DiagonalMatrix(0.48)).mean
    print(mean)
    assert not np.isnan(mean)
