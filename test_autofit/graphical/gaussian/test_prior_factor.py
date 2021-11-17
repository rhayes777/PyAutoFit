import pytest

from autofit import graphical as g
import autofit as af
from autofit.mock.mock import MockAnalysis
from test_autofit.non_linear.grid.test_optimizer_grid_search import MockOptimizer


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


def test_optimise(model):
    optimizer = MockOptimizer()
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
