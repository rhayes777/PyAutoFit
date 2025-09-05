import autofit as af
from autofit import VisualiseGraph
from autofit.mock import MockAnalysis


def test():
    model_1 = af.Model(af.Gaussian)
    analysis_factor_1 = af.AnalysisFactor(
        prior_model=model_1,
        analysis=MockAnalysis(),
    )

    model_2 = af.Model(af.Gaussian)
    analysis_factor_2 = af.AnalysisFactor(
        prior_model=model_2,
        analysis=MockAnalysis(),
    )

    model_3 = af.Model(
        af.Gaussian,
        centre=model_1.centre + model_2.centre,
    )
    analysis_factor_3 = af.AnalysisFactor(
        prior_model=model_3,
        analysis=MockAnalysis(),
    )

    factor_graph = af.FactorGraphModel(
        analysis_factor_1,
        analysis_factor_2,
        analysis_factor_3,
    )

    print(factor_graph.info)

    VisualiseGraph(factor_graph.prior_model).save("graph.html")
