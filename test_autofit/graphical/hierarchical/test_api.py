import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(
    name="hierarchical_factor"
)
def make_hierarchical_factor():
    return g.HierarchicalFactor(
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


def test_priors(
        hierarchical_factor
):
    assert len(hierarchical_factor.priors) == 2


def test_factors(
        hierarchical_factor
):
    hierarchical_factor.add_sampled_variable(
        af.GaussianPrior(100, 10)
    )
    assert len(hierarchical_factor.factors) == 1
