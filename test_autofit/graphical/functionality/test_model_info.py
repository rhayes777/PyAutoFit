import pytest

import autofit as af
from autofit import graphical as g


@pytest.fixture(
    name="analysis_factor"
)
def make_analysis_factor():
    return g.AnalysisFactor(
        prior_model=af.PriorModel(
            af.Gaussian
        ),
        analysis=af.m.MockAnalysis(),
        name="AnalysisFactor0"
    )


@pytest.fixture(
    name="info"
)
def make_info():
    return """AnalysisFactor0

centre                                                                                    UniformPrior, lower_limit = 0.0, upper_limit = 1.0
normalization                                                                             UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma                                                                                     UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def test_analysis_factor(
        analysis_factor,
        info
):
    assert analysis_factor.info == info
