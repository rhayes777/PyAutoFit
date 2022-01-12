from autofit.graphical import AnalysisFactor, PriorFactor


def test_non_trivial_results(
        non_trivial_model
):
    results_text = non_trivial_model.graph.make_results_text(
        non_trivial_model.mean_field_approximation()
    )
    assert results_text == """PriorFactors

PriorFactor0 (AnalysisFactor1.sigma)                                                      0.5
PriorFactor1 (AnalysisFactor1.normalization)                                              0.5
PriorFactor2 (AnalysisFactor0.centre, AnalysisFactor1.centre)                             0.5
PriorFactor3 (AnalysisFactor0.sigma)                                                      0.5
PriorFactor4 (AnalysisFactor0.normalization)                                              0.5

AnalysisFactors

AnalysisFactor0

centre (AnalysisFactor1.centre, PriorFactor2)                                             0.5
normalization (PriorFactor4)                                                              0.5
sigma (PriorFactor3)                                                                      0.5

AnalysisFactor1

centre (AnalysisFactor0.centre, PriorFactor2)                                             0.5
normalization (PriorFactor1)                                                              0.5
sigma (PriorFactor0)                                                                      0.5"""


def test_non_trivial_info(
        non_trivial_model
):
    info = non_trivial_model.graph.info
    assert info == """PriorFactors

PriorFactor0 (AnalysisFactor1.sigma)                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor1 (AnalysisFactor1.normalization)                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor2 (AnalysisFactor0.centre, AnalysisFactor1.centre)                             UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor3 (AnalysisFactor0.sigma)                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor4 (AnalysisFactor0.normalization)                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

centre (AnalysisFactor1.centre, PriorFactor2)                                             UniformPrior, lower_limit = 0.0, upper_limit = 1.0
normalization (PriorFactor4)                                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma (PriorFactor3)                                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactor1

centre (AnalysisFactor0.centre, PriorFactor2)                                             UniformPrior, lower_limit = 0.0, upper_limit = 1.0
normalization (PriorFactor1)                                                              UniformPrior, lower_limit = 0.0, upper_limit = 1.0
sigma (PriorFactor0)                                                                      UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def test_factors_grouped_by_type(
        factor_graph
):
    factors_by_type = factor_graph.factors_by_type()

    assert len(factors_by_type) == 2
    assert len(factors_by_type[AnalysisFactor]) == 2
    assert len(factors_by_type[PriorFactor]) == 2


def test_make_results_text(
        factor_graph,
        factor_graph_model
):
    results_text = factor_graph.make_results_text(
        factor_graph_model.mean_field_approximation()
    )
    assert results_text == """PriorFactors

PriorFactor0 (AnalysisFactor1.one)                                                        0.5
PriorFactor1 (AnalysisFactor0.one)                                                        0.5

AnalysisFactors

AnalysisFactor0

one (PriorFactor1)                                                                        0.5

AnalysisFactor1

one (PriorFactor0)                                                                        0.5"""


def test_info_for_prior_factor(
        declarative_graph_output,
        prior_factor
):
    assert declarative_graph_output.info_for_prior_factor(
        prior_factor
    ) == "PriorFactor0 (AnalysisFactor1.one)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0"


def test_info_for_analysis_factor(
        declarative_graph_output,
        analysis_factor
):
    info = declarative_graph_output.info_for_analysis_factor(
        analysis_factor
    )
    assert info == """AnalysisFactor0

one (PriorFactor1)                                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""


def test_related_factors(
        factor_graph,
        prior_factor
):
    assert len(factor_graph.related_factors(
        list(prior_factor.variables)[0]
    )) == 2


def test_graph_info(
        factor_graph
):
    info = factor_graph.info
    assert info == """PriorFactors

PriorFactor0 (AnalysisFactor1.one)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0
PriorFactor1 (AnalysisFactor0.one)                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactors

AnalysisFactor0

one (PriorFactor1)                                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0

AnalysisFactor1

one (PriorFactor0)                                                                        UniformPrior, lower_limit = 0.0, upper_limit = 1.0"""
