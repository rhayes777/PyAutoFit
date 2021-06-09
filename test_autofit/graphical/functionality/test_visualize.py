import autofit as af
from autofit import graphical as g
from autofit.mock.mock import MockAnalysis


def test_visualize():
    analysis_0 = MockAnalysis()
    analysis_1 = MockAnalysis()
    analysis_2 = MockAnalysis()

    gaussian_0 = af.Model(af.Gaussian)
    gaussian_1 = af.Model(af.Gaussian)
    gaussian_2 = af.Model(af.Gaussian)

    analysis_factor_0 = g.AnalysisFactor(
        prior_model=gaussian_0,
        analysis=analysis_0
    )
    analysis_factor_1 = g.AnalysisFactor(
        prior_model=gaussian_1,
        analysis=analysis_1
    )
    analysis_factor_2 = g.AnalysisFactor(
        prior_model=gaussian_2,
        analysis=analysis_2
    )
