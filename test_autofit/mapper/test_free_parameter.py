import pytest

import autofit as af
from autofit.non_linear.analysis.analysis import FreeParameterAnalysis


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(af.Gaussian)


def test_copy():
    model = af.Model(af.Gaussian)
    copy = model.copy()

    collection = af.Collection(model, copy)

    assert collection.prior_count == model.prior_count


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 1.0


def test_analyses_example():
    model = af.Model(af.Gaussian)
    analyses = []

    for prior, image in [
        (af.UniformPrior(), 0),
        (af.UniformPrior(), 1),
    ]:
        copy = model.copy()
        copy.centre = prior
        analyses.append(
            Analysis(

            )
        )


@pytest.fixture(
    name="combined_analysis"
)
def make_combined_analysis(model):
    return (Analysis() + Analysis()).set_free_parameter(
        model.centre
    )


def test_add_free_parameter(
        combined_analysis
):
    assert isinstance(
        combined_analysis,
        FreeParameterAnalysis
    )


def test_modify_model(
        model,
        combined_analysis
):
    modified = combined_analysis.modify_model(model)
    assert isinstance(
        modified,
        af.Collection
    )
