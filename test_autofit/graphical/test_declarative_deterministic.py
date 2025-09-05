import autofit as af
from autofit import VisualiseGraph
from autofit.mock import MockAnalysis
import numpy as np


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

    model_3 = af.Collection(
        2 * np.sqrt(2 * np.log(2)) * model_1.sigma,
        2 * np.sqrt(2 * np.log(2)) * model_2.sigma,
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
