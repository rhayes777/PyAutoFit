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
            sigma=1
        )
    )
    factor.add_drawn_variable(
        prior
    )
    return factor
