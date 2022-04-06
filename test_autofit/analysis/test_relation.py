import pytest

from autofit.non_linear.analysis.model_analysis import CombinedModelAnalysis


@pytest.fixture(
    name="model_analysis"
)
def make_model_analysis(Analysis, model):
    return Analysis().with_model(model)


def test_analysis_model(model_analysis, model):
    assert model_analysis.modify_model(model) is model


@pytest.fixture(
    name="combined_model_analysis"
)
def make_combined_model_analysis(
        model_analysis
):
    return model_analysis + model_analysis


def test_combined_model_analysis(
        combined_model_analysis
):
    assert isinstance(
        combined_model_analysis,
        CombinedModelAnalysis
    )


def test_modify(
        combined_model_analysis,
        model
):
    modified = combined_model_analysis.modify_model(
        model
    )
    first, second = modified
    assert first is model
    assert second is model
