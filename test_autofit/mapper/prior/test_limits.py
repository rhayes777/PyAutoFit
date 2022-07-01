import pytest

import numpy as np

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


def test_optional(prior):
    prior.value_for(
        0.0,
        ignore_prior_limits=True
    )


@pytest.fixture(
    name="model"
)
def make_model(prior):
    return af.Model(
        af.Gaussian,
        centre=prior
    )


def test_vector_from_unit_vector(model):
    with pytest.raises(
            PriorLimitException
    ):
        model.vector_from_unit_vector([
            0, 0, 0
        ])


def test_vector_ignore_limits(model):
    model.vector_from_unit_vector(
        [0, 0, 0],
        ignore_prior_limits=True
    )


@pytest.mark.parametrize(
    "prior",
    [
        af.LogUniformPrior(),
        af.UniformPrior(),
        af.GaussianPrior(
            mean=0,
            sigma=1,
            lower_limit=0.0,
            upper_limit=1.0,
        )
    ]
)
@pytest.mark.parametrize(
    "value",
    [-1.0, 2.0]
)
def test_all_priors(
        prior,
        value
):
    with pytest.raises(
            PriorLimitException
    ):
        prior.value_for(value)

    prior.value_for(
        value,
        ignore_prior_limits=True
    )


@pytest.mark.parametrize(
    "value",
    np.arange(0, 1, 0.1)
)
def test_invert_limits(value):
    value = float(value)
    prior = af.GaussianPrior(
        mean=1.0,
        sigma=2.0,
    )
    assert prior.message.cdf(
        prior.value_for(value)
    ) == pytest.approx(value)


def test_unit_limits():
    prior = af.GaussianPrior(
        mean=1.0,
        sigma=2.0,
        lower_limit=-10,
        upper_limit=5,
    )
    EPSILON = 0.00001
    assert prior.value_for(
        prior.lower_unit_limit
    )
    assert prior.value_for(
        prior.upper_unit_limit - EPSILON
    )

    with pytest.raises(PriorLimitException):
        prior.value_for(
            prior.lower_unit_limit - EPSILON
        )
    with pytest.raises(PriorLimitException):
        prior.value_for(
            prior.upper_unit_limit + EPSILON
        )
