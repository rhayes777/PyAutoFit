import itertools

import pytest

import autofit as af
from autofit import graphical as g
from autofit.graphical import AnalysisFactor
from autofit.graphical.declarative.graph import GraphInfoFormatter
from autofit.tools.namer import namer


@pytest.fixture(
    autouse=True
)
def reset_namer():
    namer.reset()
    yield
    namer.reset()
    af.ModelObject._ids = itertools.count()


@pytest.fixture(
    name="factor_graph_model"
)
def make_factor_graph_model():
    model_factor_1 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        af.m.MockAnalysis()
    )
    model_factor_2 = g.AnalysisFactor(
        af.Collection(
            one=af.UniformPrior()
        ),
        af.m.MockAnalysis()
    )

    return g.FactorGraphModel(
        model_factor_1,
        model_factor_2
    )


@pytest.fixture(
    name="non_trivial_model"
)
def make_non_trivial_model():
    one = af.Model(af.Gaussian)
    two = af.Model(af.Gaussian)

    one.centre = two.centre

    model_factor_1 = g.AnalysisFactor(
        one,
        af.m.MockAnalysis()
    )
    model_factor_2 = g.AnalysisFactor(
        two,
        af.m.MockAnalysis()
    )

    return g.FactorGraphModel(
        model_factor_1,
        model_factor_2
    )


@pytest.fixture(
    name="factor_graph"
)
def make_factor_graph(
        factor_graph_model
):
    return factor_graph_model.graph


def test_factors_with_type(
        factor_graph
):
    factor_type = AnalysisFactor
    factors = factor_graph._factors_with_type(
        factor_type
    )
    assert len(factors) == 2
    for factor in factors:
        assert isinstance(
            factor,
            AnalysisFactor
        )


@pytest.fixture(
    name="prior_factor"
)
def make_prior_factor(
        factor_graph
):
    return factor_graph.prior_factors[0]


@pytest.fixture(
    name="analysis_factor"
)
def make_analysis_factor(
        factor_graph
):
    return factor_graph.analysis_factors[0]


@pytest.fixture(
    name="declarative_graph_output"
)
def make_declarative_graph_output(
        factor_graph
):
    return GraphInfoFormatter(
        factor_graph
    )
