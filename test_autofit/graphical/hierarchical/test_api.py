import autofit as af
from autofit import graphical as g


def test_hierarchical_factor_api():
    hierarchical_factor = g.HierarchicalFactor(
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
    assert len(hierarchical_factor.priors) == 2

    hierarchical_factor.add_sampled_variable(
        af.GaussianPrior(100, 10)
    )
    assert len(hierarchical_factor.priors) == 3
