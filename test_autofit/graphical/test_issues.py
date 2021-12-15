import pytest

import autofit as af
from autofit.exc import PriorLimitException


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.GaussianPrior(
        mean=3.0,
        sigma=5.0,
        lower_limit=0.0
    )


def test_intrinsic_lower_limit(prior):
    with pytest.raises(
            PriorLimitException
    ):
        prior.value_for(0.0)


def test_prior_factor(prior):
    prior.factor(1.0)

    with pytest.raises(
            PriorLimitException
    ):
        prior.factor(-1.0)


def test_ignore_prior_limits(prior):
    prior.value_for(
        0.0,
        ignore_prior_limits=True
    )
