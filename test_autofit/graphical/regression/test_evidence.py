import pytest

import autofit as af
from autofit import messages as m


@pytest.mark.parametrize(
    "message",
    [
        af.UniformPrior().message,
        af.GaussianPrior(
            mean=1.0,
            sigma=2.0
        ).message,
        m.NormalMessage(
            mean=0.0,
            sigma=1.0
        )
    ]
)
def test_log_normalisation(message):
    assert m.AbstractMessage.log_normalisation(message) > 0.0
