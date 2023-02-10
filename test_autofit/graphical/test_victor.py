import numpy as np

import autofit as af


def _test_divide_uniform():
    """
    Currently fails because division poorly defined for identical Normal Messages
    """
    prior = af.UniformPrior(
        lower_limit=0.2,
        upper_limit=0.8
    )

    result = prior / af.UniformPrior()
    assert not np.isnan(result.lower_limit)
