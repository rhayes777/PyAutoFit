import autofit as af
from autofit import graphical as g
from autofit.mock.mock import MockAnalysis


def test_analysis_factor():
    factor_model = g.AnalysisFactor(
        prior_model=af.PriorModel(
            af.Gaussian
        ),
        analysis=MockAnalysis()
    )
    assert factor_model.info == """AnalysisFactor0

centre                                                                                    UniformPrior, lower_limit = 0.0, upper_limit = 1.0
intensity                                                                                 UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma                                                                                     UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""
