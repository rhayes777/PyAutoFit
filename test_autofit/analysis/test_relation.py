import pytest

from autofit.non_linear.analysis.model_analysis import CombinedModelAnalysis


@pytest.fixture(
    name="model_analysis"
)
def make_model_analysis(Analysis, model):
    return Analysis().with_model(model)


def test_analysis_model(model_analysis, model):
    assert model_analysis.modify_model(model) is model


def test_combined_model_analysis(
        model_analysis
):
    assert isinstance(
        model_analysis + model_analysis,
        CombinedModelAnalysis
    )
