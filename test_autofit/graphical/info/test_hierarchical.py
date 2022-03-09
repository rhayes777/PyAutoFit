import pytest

import autofit as af
from autofit import graphical as g
from autofit.graphical.declarative.graph import GraphInfoFormatter


@pytest.fixture(name="model_factor_1")
def make_model_factor_1():
    return g.AnalysisFactor(af.Model(af.Gaussian), af.m.MockAnalysis())


@pytest.fixture(name="model_factor_2")
def make_model_factor_2():
    return g.AnalysisFactor(af.Model(af.Gaussian), af.m.MockAnalysis())


@pytest.fixture(name="hierarchical_factor")
def make_hierarchical_factor(
        model_factor_1,
        model_factor_2,
):
    hierarchical_factor = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=100, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=5),
    )

    hierarchical_factor.add_drawn_variable(model_factor_1.centre)
    hierarchical_factor.add_drawn_variable(model_factor_2.centre)
    return hierarchical_factor


@pytest.fixture(name="hierarchical_model")
def make_factor_graph_model(
        model_factor_1,
        model_factor_2,
        hierarchical_factor,
):
    return g.FactorGraphModel(model_factor_1, model_factor_2, hierarchical_factor)


def test_hierarchical_factors(graph, hierarchical_factor):
    assert graph.hierarchical_factors == [hierarchical_factor]


@pytest.fixture(name="graph")
def make_graph(hierarchical_model):
    return hierarchical_model.graph


def test_info_for_hierarchical_factor(graph):
    info = GraphInfoFormatter(graph).info_for_hierarchical_factor(
        graph.hierarchical_factors[0]
    )
    assert (
            info
            == """HierarchicalFactor0

mean (HierarchicalFactor0, PriorFactor1)                                                  GaussianPrior, mean = 100, sigma = 10
sigma (HierarchicalFactor0, PriorFactor0)                                                 GaussianPrior, mean = 10, sigma = 5

Drawn Variables

AnalysisFactor0.centre, PriorFactor7                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0
AnalysisFactor1.centre, PriorFactor4                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""
    )


def test_graph_info(graph):
    info = graph.info
    assert (
            info
            == """PriorFactors

PriorFactor0 (HierarchicalFactor0)                                                        GaussianPrior, mean = 10, sigma = 5
PriorFactor1 (HierarchicalFactor0)                                                        GaussianPrior, mean = 100, sigma = 10
PriorFactor2 (AnalysisFactor1.sigma)                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor3 (AnalysisFactor1.normalization)                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor4 (AnalysisFactor1.centre, HierarchicalFactor0)                                UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor5 (AnalysisFactor0.sigma)                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor6 (AnalysisFactor0.normalization)                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor7 (AnalysisFactor0.centre, HierarchicalFactor0)                                UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

centre (HierarchicalFactor0, PriorFactor7)                                                UniformPrior, lower_limit = 0.0, upper_limit = 1.0
normalization (PriorFactor6)                                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma (PriorFactor5)                                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactor1

centre (HierarchicalFactor0, PriorFactor4)                                                UniformPrior, lower_limit = 0.0, upper_limit = 1.0
normalization (PriorFactor3)                                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma (PriorFactor2)                                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0

HierarchicalFactors

HierarchicalFactor0

mean (HierarchicalFactor0, PriorFactor1)                                                  GaussianPrior, mean = 100, sigma = 10
sigma (HierarchicalFactor0, PriorFactor0)                                                 GaussianPrior, mean = 10, sigma = 5

Drawn Variables

AnalysisFactor0.centre, PriorFactor7                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0
AnalysisFactor1.centre, PriorFactor4                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""
    )
