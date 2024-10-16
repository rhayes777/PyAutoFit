import pytest

import autofit as af
from autofit.non_linear.analysis import FreeParameterAnalysis
from autofit.non_linear.mock.mock_search import MockMLE


def test_copy():
    model = af.Model(af.Gaussian)
    copy = model.copy()

    collection = af.Collection(model, copy)

    assert collection.prior_count == model.prior_count


def test_log_likelihood(modified, combined_analysis):
    assert (
        combined_analysis.log_likelihood_function(
            modified.instance_from_prior_medians()
        )
        == 2
    )


def test_analyses_example(Analysis):
    model = af.Model(af.Gaussian)
    analyses = []

    for prior, image in [
        (af.UniformPrior(), 0),
        (af.UniformPrior(), 1),
    ]:
        copy = model.copy()
        copy.centre = prior
        analyses.append(Analysis())


@pytest.fixture(name="combined_analysis")
def make_combined_analysis(model, Analysis):
    return (Analysis() + Analysis()).with_free_parameters(model.centre)


def test_override_specific_free_parameter(model, combined_analysis):
    combined_analysis[0][model.centre] = 2

    new_model = combined_analysis.modify_model(model)
    assert new_model[0].centre == 2
    assert new_model[1].centre != 2


def test_override_multiple(model, combined_analysis):
    combined_analysis[0][model.centre] = 2
    combined_analysis[1][model.centre] = 3

    new_model = combined_analysis.modify_model(model)
    assert new_model[0].centre == 2
    assert new_model[1].centre == 3


def test_override_multiple_one_analysis(model, combined_analysis):
    combined_analysis[0][model.centre] = 2
    combined_analysis[0][model.sigma] = 3

    new_model = combined_analysis.modify_model(model)
    assert new_model[0].centre == 2
    assert new_model[0].sigma == 3


def test_complex_path(Analysis):
    model = af.Collection(
        collection=af.Collection(
            gaussian=af.Model(af.Gaussian),
        )
    )
    combined_analysis = (Analysis() + Analysis()).with_free_parameters(
        model.collection.gaussian
    )

    combined_analysis[0][model.collection.gaussian.centre] = 2
    combined_analysis[0][model.collection.gaussian.sigma] = 3

    new_model = combined_analysis.modify_model(model)
    assert new_model[0].collection.gaussian.centre == 2
    assert new_model[0].collection.gaussian.sigma == 3


def test_override_model(model, combined_analysis):
    new_model = af.Model(af.Gaussian, centre=2)
    combined_analysis[0][model] = new_model

    new_model = combined_analysis.modify_model(model)
    assert new_model[0].centre == 2
    assert new_model[1].centre != 2


def test_override_child_model(Analysis):
    model = af.Collection(gaussian=af.Gaussian)
    combined_analysis = (Analysis() + Analysis()).with_free_parameters(
        model.gaussian.centre
    )
    combined_analysis[0][model.gaussian] = af.Model(af.Gaussian, centre=2)

    new_model = combined_analysis.modify_model(model)
    assert new_model[0].gaussian.centre == 2


def test_multiple_free_parameters(model, Analysis):
    combined_analysis = (Analysis() + Analysis()).with_free_parameters(
        model.centre, model.sigma
    )
    first, second = combined_analysis.modify_model(model)
    assert first.centre is not second.centre
    assert first.sigma is not second.sigma


def test_add_free_parameter(combined_analysis):
    assert isinstance(combined_analysis, FreeParameterAnalysis)


@pytest.fixture(name="modified")
def make_modified(model, combined_analysis):
    return combined_analysis.modify_model(model)


def test_modify_model(modified):
    assert isinstance(modified, af.Collection)
    assert len(modified) == 2


def test_modified_models(modified):
    first, second = modified

    assert isinstance(first.sigma, af.Prior)
    assert first.sigma == second.sigma
    assert first.centre != second.centre


@pytest.fixture(name="result")
def make_result(
    combined_analysis,
    model,
):
    search = MockMLE()
    return search.fit(model, combined_analysis)


@pytest.fixture(autouse=True)
def do_remove_output(remove_output):
    yield
    remove_output()


def test_tuple_prior(model, Analysis):
    model.centre = af.TuplePrior(centre_0=af.UniformPrior())
    combined = (Analysis() + Analysis()).with_free_parameters(model.centre)

    first, second = combined.modify_model(model)
    assert first.centre.centre_0 != second.centre.centre_0


def test_prior_model(model, Analysis):
    model = af.Collection(model=model)
    combined = (Analysis() + Analysis()).with_free_parameters(model.model)
    modified = combined.modify_model(model)
    first = modified[0].model
    second = modified[1].model

    assert first is not second
    assert first != second
    assert first.centre != second.centre


def test_split_samples(modified):
    samples = af.Samples(
        modified,
        af.Sample.from_lists(modified, [[1, 2, 3, 4]], [1], [1], [1]),
    )

    combined = samples.max_log_likelihood()

    first = samples.subsamples(modified[0])
    second = samples.subsamples(modified[1])

    assert first.max_log_likelihood().centre == combined[0].centre
    assert second.max_log_likelihood().centre == combined[1].centre
