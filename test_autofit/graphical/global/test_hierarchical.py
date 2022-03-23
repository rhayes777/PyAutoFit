import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(
    name="graph"
)
def make_graph(
        model_factor,
        model_factor_2,
):
    hierarchical_factor = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(
            mean=0.5,
            sigma=0.1
        ),
        sigma=af.GaussianPrior(
            mean=1.0,
            sigma=0.01
        )
    )

    hierarchical_factor.add_drawn_variable(
        model_factor.one
    )
    hierarchical_factor.add_drawn_variable(
        model_factor_2.one
    )

    return g.FactorGraphModel(
        hierarchical_factor,
        model_factor,
        model_factor
    )


def test_info(graph):
    assert graph.global_prior_model.info == """PriorFactors

PriorFactor0 (HierarchicalFactor0)                                                        GaussianPrior, mean = 1.0, sigma = 0.01
PriorFactor1 (HierarchicalFactor0)                                                        GaussianPrior, mean = 0.5, sigma = 0.1
PriorFactor2 (HierarchicalFactor0)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor3 (AnalysisFactor0.one, HierarchicalFactor0)                                   UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

one (HierarchicalFactor0, PriorFactor3)                                                   UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactor0

one (HierarchicalFactor0, PriorFactor3)                                                   UniformPrior, lower_limit = 0.0, upper_limit = 1.0

HierarchicalFactors

HierarchicalFactor0

mean (HierarchicalFactor0, PriorFactor1)                                                  GaussianPrior, mean = 0.5, sigma = 0.1
sigma (HierarchicalFactor0, PriorFactor0)                                                 GaussianPrior, mean = 1.0, sigma = 0.01

Drawn Variables

AnalysisFactor0.one, PriorFactor3                                                         UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor2                                                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def test_instance(graph):
    model = graph.global_prior_model
    assert model.prior_count == 4
    instance = model.instance_from_unit_vector(
        [0.1, 0.2, 0.3, 0.4]
    )
    dist_model_1 = instance[0].distribution_model
    assert isinstance(
        dist_model_1,
        af.GaussianPrior
    )
    assert instance[0].drawn_prior == 0.1

    dist_model_2 = instance[0].distribution_model
    assert isinstance(
        dist_model_2,
        af.GaussianPrior
    )
    assert instance[1].drawn_prior == 0.2

    assert dist_model_1 == dist_model_2
