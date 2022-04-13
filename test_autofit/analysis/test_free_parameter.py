import pytest

import autofit as af
from autofit.non_linear.analysis import FreeParameterAnalysis
from autofit.non_linear.mock.mock_search import MockOptimizer


def test_copy():
    model = af.Model(af.Gaussian)
    copy = model.copy()

    collection = af.Collection(model, copy)

    assert collection.prior_count == model.prior_count


def test_log_likelihood(
        modified,
        combined_analysis
):
    assert combined_analysis.log_likelihood_function(
        modified.instance_from_prior_medians()
    ) == 2


def test_analyses_example(Analysis):
    model = af.Model(af.Gaussian)
    analyses = []

    for prior, image in [
        (af.UniformPrior(), 0),
        (af.UniformPrior(), 1),
    ]:
        copy = model.copy()
        copy.centre = prior
        analyses.append(
            Analysis()
        )


@pytest.fixture(
    name="combined_analysis"
)
def make_combined_analysis(model, Analysis):
    return (Analysis() + Analysis()).with_free_parameters(
        model.centre
    )


def test_multiple_free_parameters(model, Analysis):
    combined_analysis = (Analysis() + Analysis()).with_free_parameters(
        model.centre,
        model.sigma
    )
    first, second = combined_analysis.modify_model(model)
    assert first.centre is not second.centre
    assert first.sigma is not second.sigma


def test_add_free_parameter(
        combined_analysis
):
    assert isinstance(
        combined_analysis,
        FreeParameterAnalysis
    )


@pytest.fixture(
    name="modified"
)
def make_modified(
        model,
        combined_analysis
):
    return combined_analysis.modify_model(model)


def test_modify_model(
        modified
):
    assert isinstance(modified, af.Collection)
    assert len(modified) == 2


def test_modified_models(
        modified
):
    first, second = modified

    assert isinstance(
        first.sigma,
        af.Prior
    )
    assert first.sigma == second.sigma
    assert first.centre != second.centre


@pytest.fixture(
    name="result"
)
def make_result(
        combined_analysis,
        model,
):
    optimizer = MockOptimizer()
    return optimizer.fit(
        model,
        combined_analysis
    )


def test_result_type(result, Result):
    assert isinstance(result, Result)

    for result_ in result:
        assert isinstance(result_, Result)


def test_integration(result):
    result_1, result_2 = result

    assert result_1._model.centre is not result_2._model.centre
    assert result_1._model.sigma is result_2._model.sigma


def test_tuple_prior(model, Analysis):
    model.centre = af.TuplePrior(
        centre_0=af.UniformPrior()
    )
    combined = (Analysis() + Analysis()).with_free_parameters(
        model.centre
    )

    first, second = combined.modify_model(model)
    assert first.centre.centre_0 != second.centre.centre_0


def test_prior_model(model, Analysis):
    model = af.Collection(
        model=model
    )
    combined = (Analysis() + Analysis()).with_free_parameters(
        model.model
    )
    modified = combined.modify_model(model)
    first = modified[0].model
    second = modified[1].model

    assert first is not second
    assert first != second
    assert first.centre != second.centre


def test_split_samples(modified):
    samples = af.Samples(
        modified,
        af.Sample.from_lists(
            modified,
            [[1, 2, 3, 4]],
            [1], [1], [1]
        ),
    )

    combined = samples.max_log_likelihood_instance

    first = samples.subsamples(modified[0])
    second = samples.subsamples(modified[1])

    assert first.max_log_likelihood_instance.centre == combined[0].centre
    assert second.max_log_likelihood_instance.centre == combined[1].centre
