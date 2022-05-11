import pytest

import autofit as af
from autofit.messages import AbstractMessage


def test_gaussian_prior():
    prior = af.GaussianPrior(
        mean=0.0,
        sigma=1.0,
    )

    assert not isinstance(
        prior,
        AbstractMessage
    )
    assert isinstance(
        prior.message,
        AbstractMessage
    )

    assert prior.value_for(0.5) == 0.0


@pytest.mark.parametrize(
    "cls",
    [
        af.UniformPrior,
        af.LogUniformPrior,
    ]
)
def test_other_priors(cls):
    prior = cls()

    assert not isinstance(
        prior,
        AbstractMessage
    )
    assert isinstance(
        prior.value_for(0.5),
        float
    )
