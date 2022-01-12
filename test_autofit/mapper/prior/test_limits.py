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
