import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(
    name="prior"
)
def make_prior(model_gaussian_x1):
    return model_gaussian_x1.priors[0]


def test_collection_prior_model(
        prior
):
    assert af.Collection(prior).prior_count == 1


@pytest.fixture(
    name="prior_factor"
)
def make_prior_factor(prior):
    return g.PriorFactor(
        prior
    )


def test_prior_factor(
        prior_factor
):
    assert prior_factor.prior_model.prior_count == 1


def test_log_likelihood_function(
        prior_factor
):
    instance = prior_factor.prior_model.instance_from_prior_medians()
    assert prior_factor.log_likelihood_function(
        instance
    ) == pytest.approx(-0.9189385332046727, 1.0e-8)


def test_optimise(model_gaussian_x1, prior):
    optimizer = af.DynestyStatic(
        maxcall=10
    )
    analysis = af.m.MockAnalysis()
    factor = g.AnalysisFactor(
        model_gaussian_x1,
        analysis
    )
    prior_factor = factor.prior_factors[0]
    result, status = optimizer.optimise(
        prior_factor,
        factor.mean_field_approximation()
    )

    assert status

    optimized_mean = list(result.mean_field.values())[0].mean
    assert optimized_mean == pytest.approx(
        prior.mean,
        rel=0.1
    )
