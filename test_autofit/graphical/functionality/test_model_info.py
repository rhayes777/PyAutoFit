import pytest

import autofit as af
from autofit import graphical as g
from autofit.mock.mock import MockAnalysis


@pytest.fixture(
    name="analysis_factor"
)
def make_analysis_factor():
    return g.AnalysisFactor(
        prior_model=af.PriorModel(
            af.Gaussian
        ),
        analysis=MockAnalysis(),
        name="AnalysisFactor0"
    )


@pytest.fixture(
    name="info"
)
def make_info():
    return """AnalysisFactor0

centre                                                                                    UniformPrior, lower_limit = 0.0, upper_limit = 1.0
intensity                                                                                 UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma                                                                                     UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def test_analysis_factor(
        analysis_factor,
        info
):
    assert analysis_factor.info == info


def test_graph(
        analysis_factor,
        info
):
    graph = g.FactorGraphModel(
        analysis_factor,
        analysis_factor,
        name="FactorGraphModel0"
    )

    assert graph.info == f"FactorGraphModel0\n\n{info}\n\n{info}"
