import itertools

import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(name="graph")
def make_graph(
    model_factor,
    model_factor_2,
):
    hierarchical_factor = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=0.5, sigma=0.1),
        sigma=af.GaussianPrior(mean=1.0, sigma=0.01),
    )

    hierarchical_factor.add_drawn_variable(model_factor.one)
    hierarchical_factor.add_drawn_variable(model_factor_2.one)

    return g.FactorGraphModel(hierarchical_factor, model_factor, model_factor)


@pytest.fixture(name="model")
def make_model(graph):
    return graph.global_prior_model


@pytest.fixture(autouse=True)
def reset_ids():
    af.Prior._ids = itertools.count()


def test_model_info(model):
    print(model.info)
    assert (
        model.info
        == """Total Free Parameters = 4

model                                                                           GlobalPriorModel (N=4)
    0 - 1                                                                       Collection (N=3)
        distribution_model                                                      GaussianPrior (N=2)
    2 - 3                                                                       Collection (N=1)

0 - 1
    distribution_model
        mean                                                                    GaussianPrior [2], mean = 0.5, sigma = 0.1
        sigma                                                                   GaussianPrior [3], mean = 1.0, sigma = 0.01
0
    drawn_prior                                                                 UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
1
    drawn_prior                                                                 UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
2 - 3
    one                                                                         UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0"""
    )


def test_graph_info(model):
    assert (
        model.graph_info
        == """PriorFactors

PriorFactor0 (HierarchicalFactor0)                                                        GaussianPrior [3], mean = 1.0, sigma = 0.01
PriorFactor1 (HierarchicalFactor0)                                                        GaussianPrior [2], mean = 0.5, sigma = 0.1
PriorFactor2 (HierarchicalFactor0)                                                        UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
PriorFactor3 (AnalysisFactor0.one, HierarchicalFactor0)                                   UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

one (HierarchicalFactor0, PriorFactor3)                                                   UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0

AnalysisFactor0

one (HierarchicalFactor0, PriorFactor3)                                                   UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0

HierarchicalFactors

HierarchicalFactor0

mean (HierarchicalFactor0, PriorFactor1)                                                  GaussianPrior [2], mean = 0.5, sigma = 0.1
sigma (HierarchicalFactor0, PriorFactor0)                                                 GaussianPrior [3], mean = 1.0, sigma = 0.01

Drawn Variables

AnalysisFactor0.one, PriorFactor3                                                         UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
PriorFactor2                                                                              UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0"""
    )


def test_instance(model):
    assert model.prior_count == 4
    instance = model.instance_from_unit_vector([0.1, 0.2, 0.3, 0.4])
    dist_model_1 = instance[0].distribution_model
    assert isinstance(dist_model_1, af.GaussianPrior)
    assert instance[0].drawn_prior == pytest.approx(0.1)

    dist_model_2 = instance[0].distribution_model
    assert isinstance(dist_model_2, af.GaussianPrior)
    assert instance[1].drawn_prior == pytest.approx(0.2)

    assert dist_model_1 == dist_model_2


@pytest.mark.parametrize(
    "unit_vector, likelihood",
    [
        ([0.1, 0.2, 0.3, 0.5], -2.248),
        ([0.1, 0.2, 0.4, 0.8], -2.280),
        ([0.1, 0.2, 0.3, 0.3], -2.239),
    ],
)
def test_likelihood(
    graph,
    model,
    unit_vector,
    likelihood,
):
    instance = model.instance_from_unit_vector(unit_vector)
    assert graph.log_likelihood_function(instance) == pytest.approx(
        likelihood, rel=0.001
    )
