import pytest

import autofit as af
from autofit.mapper.prior.compound import MultiplePrior, SumPrior
from autofit.non_linear.analysis.free_parameter import IndexedAnalysis
from autofit.non_linear.analysis.model_analysis import CombinedModelAnalysis, ModelAnalysis


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

    for analysis in combined_model_analysis.analyses:
        assert isinstance(analysis, IndexedAnalysis)
        assert isinstance(analysis.analysis, ModelAnalysis)


def test_sum(model):
    analyses = 3 * [af.Analysis().with_model(model)]
    analysis = sum(analyses)

    for analysis_ in analysis.analyses:
        assert isinstance(analysis_, IndexedAnalysis)
        assert isinstance(analysis_.analysis, ModelAnalysis)


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


def test_default(
        combined_model_analysis,
        Analysis,
        model
):
    analysis = combined_model_analysis + Analysis()
    assert isinstance(
        analysis,
        CombinedModelAnalysis
    )
    modified = analysis.modify_model(model)

    first, second, third = modified
    assert first is model
    assert second is model
    assert third is model


def test_fit(
        combined_model_analysis,
        model
):
    assert combined_model_analysis.log_likelihood_function(
        af.Collection([model, model]).instance_from_prior_medians()
    ) == 2


def test_prior_arithmetic():
    m = af.UniformPrior()
    c = af.UniformPrior()
    mul = MultiplePrior(10, m)
    y = SumPrior(mul, c)

    assert y.prior_count == 2
    assert y.instance_from_prior_medians() == 5.5


class LinearAnalysis(af.Analysis):
    def __init__(self, value):
        self.value = value

    def log_likelihood_function(self, instance):
        return -abs(self.value - instance)


def test_integration():
    def data(x):
        return 3 * x + 5

    m = af.GaussianPrior(
        mean=3,
        sigma=1
    )
    c = af.GaussianPrior(
        mean=5,
        sigma=1
    )

    analyses = [
        LinearAnalysis(
            data(x)
        ).with_model(
            SumPrior(
                MultiplePrior(
                    x, m
                ),
                c
            )
        )
        for x in range(10)
    ]

    combined = sum(analyses)

    optimiser = af.DynestyStatic()
    result = optimiser.fit(None, combined)

    print(result)
