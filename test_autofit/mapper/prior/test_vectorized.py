import numpy as np
import pytest

import autofit as af


class MockModel:

    def __init__(self, priors_ordered_by_id):

        self.priors_ordered_by_id = priors_ordered_by_id


@pytest.mark.parametrize(
    "lower, upper, unit",
    [
        (0.0, 1.0, 0.5),
        (1.0, 3.0, 0.25),
        (-2.0, 2.0, 0.75),
    ]
)
def test__uniform_vectorized_vs_scalar(lower, upper, unit):

    prior = af.UniformPrior(lower_limit=lower, upper_limit=upper)

    # Scalar transform
    value = prior.value_for(unit=unit)

    # Vectorized transform
    model = MockModel(priors_ordered_by_id=[prior])
    vectorized = af.PriorVectorized(model=model)
    value_via_vectorized = vectorized(np.array([[unit]]))

    assert np.allclose(value, value_via_vectorized[0])