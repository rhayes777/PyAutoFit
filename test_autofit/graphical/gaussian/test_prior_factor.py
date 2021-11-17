import pytest

import autofit as af
from autofit import graphical as g
from autofit.mock.mock import MockAnalysis


@pytest.fixture(
    name="prior"
)
def make_prior(model):
    return model.priors[0]


def test_collection_prior_model(
        prior
):
    assert af.Collection(prior).prior_count == 1


def test_prior_factor(prior):
    prior_factor = g.PriorFactor(
        prior
    )
    assert prior_factor.prior_model.prior_count == 1


def test_optimise(model, prior):
    # optimizer = MockOptimizer()
    optimizer = af.DynestyStatic(
        maxcall=10
    )
    analysis = MockAnalysis()
    factor = g.AnalysisFactor(
        model,
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
