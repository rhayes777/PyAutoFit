import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(
    name="prior"
)
def make_prior():
    return af.GaussianPrior(100, 10)


@pytest.fixture(
    name="hierarchical_factor"
)
def make_hierarchical_factor(prior):
    factor = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(
            mean=100,
            sigma=10
        ),
        sigma=af.GaussianPrior(
            mean=10,
            sigma=5
        )
    )
    factor.add_sampled_variable(
        prior
    )
    return factor


def test_priors(
        hierarchical_factor
):
    assert len(hierarchical_factor.priors) == 2


def test_factors(
        hierarchical_factor
):
    assert len(hierarchical_factor.factors) == 1


def test_factor(
        hierarchical_factor,
        prior
):
    factor = hierarchical_factor.factors[0]
    assert factor.prior_model is hierarchical_factor
    assert factor.sample_prior is prior
