import autofit as af
from autofit.mock import MockAnalysis


def test():

    pass

    # model_1 = af.Model(af.ex.Gaussian)
    # analysis_factor_1 = af.AnalysisFactor(
    #     prior_model=model_1,
    #     analysis=MockAnalysis(),
    # )
    #
    # model_2 = af.Model(af.ex.Gaussian)
    # analysis_factor_2 = af.AnalysisFactor(
    #     prior_model=model_2,
    #     analysis=MockAnalysis(),
    # )
    #
    # model_3 = af.Collection(
    #     model_1.fwhm,
    #     model_2.fwhm,
    # )
    # analysis_factor_3 = af.AnalysisFactor(
    #     prior_model=model_3,
    #     analysis=MockAnalysis(),
    # )
    #
    # factor_graph = af.FactorGraphModel(
    #     analysis_factor_1,
    #     analysis_factor_2,
    #     analysis_factor_3,
    # )

#     assert (
#         factor_graph.info
#         == """PriorFactors
#
# PriorFactor0 (AnalysisFactor1.sigma, AnalysisFactor2.1.self)                              UniformPrior [5], lower_limit = 0.0, upper_limit = 1.0
# PriorFactor1 (AnalysisFactor1.normalization)                                              UniformPrior [4], lower_limit = 0.0, upper_limit = 1.0
# PriorFactor2 (AnalysisFactor1.centre)                                                     UniformPrior [3], lower_limit = 0.0, upper_limit = 1.0
# PriorFactor3 (AnalysisFactor0.sigma, AnalysisFactor2.0.self)                              UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0
# PriorFactor4 (AnalysisFactor0.normalization)                                              UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
# PriorFactor5 (AnalysisFactor0.centre)                                                     UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
#
# AnalysisFactors
#
# AnalysisFactor0
#
# centre (PriorFactor5)                                                                     UniformPrior [0], lower_limit = 0.0, upper_limit = 1.0
# normalization (PriorFactor4)                                                              UniformPrior [1], lower_limit = 0.0, upper_limit = 1.0
# sigma (AnalysisFactor2.0.self, PriorFactor3)                                              UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0
#
# AnalysisFactor1
#
# centre (PriorFactor2)                                                                     UniformPrior [3], lower_limit = 0.0, upper_limit = 1.0
# normalization (PriorFactor1)                                                              UniformPrior [4], lower_limit = 0.0, upper_limit = 1.0
# sigma (AnalysisFactor2.1.self, PriorFactor0)                                              UniformPrior [5], lower_limit = 0.0, upper_limit = 1.0
#
# AnalysisFactor2
#
# 0
#     self (AnalysisFactor0.sigma, PriorFactor3)                                            UniformPrior [2], lower_limit = 0.0, upper_limit = 1.0
# 1
#     self (AnalysisFactor1.sigma, PriorFactor0)                                            UniformPrior [5], lower_limit = 0.0, upper_limit = 1.0"""
#     )
