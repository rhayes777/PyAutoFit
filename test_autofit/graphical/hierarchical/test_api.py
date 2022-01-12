import autofit as af
from autofit import graphical as g


def test_priors(
        hierarchical_factor
):
    assert len(hierarchical_factor.priors) == 2


def test_factors(
        hierarchical_factor
):
    assert len(hierarchical_factor.factors) == 1

    hierarchical_factor.add_drawn_variable(
        af.UniformPrior(
            lower_limit=0.0,
            upper_limit=1.0
        )
    )
    assert len(hierarchical_factor.factors) == 2


def test_factor(
        hierarchical_factor,
        prior
):
    factor = hierarchical_factor.factors[0]
    assert factor.distribution_model is hierarchical_factor
    assert factor.drawn_prior is prior


def test_graph(
        hierarchical_factor
):
    graph = g.FactorGraphModel(
        hierarchical_factor
    )
    assert len(graph.model_factors) == 1

    hierarchical_factor.add_drawn_variable(
        af.UniformPrior(
            lower_limit=0.0,
            upper_limit=1.0
        )
    )
    assert len(graph.model_factors) == 2
