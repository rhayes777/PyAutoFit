import pytest

import autofit as af
from autofit.exc import PriorLimitException


def test_intrinsic_lower_limit():
    prior = af.GaussianPrior(
        mean=3.0,
        sigma=5.0,
        lower_limit=0.0
    )

    with pytest.raises(
            PriorLimitException
    ):
        prior.value_for(0.0)


def test_prior_factor():
    prior = af.GaussianPrior(
        mean=3.0,
        sigma=5.0,
        lower_limit=0.0
    )

    prior.factor(1.0)

    with pytest.raises(
            PriorLimitException
    ):
        prior.factor(-1.0)
