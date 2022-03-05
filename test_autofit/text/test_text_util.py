from os import path
import pytest

import autofit as af

from autofit.text import text_util

text_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files", "samples")


@pytest.fixture(name="model")
def make_model():
    return af.ModelMapper(mock_class=af.m.MockClassx2)


@pytest.fixture(name="samples")
def make_samples(model):
    parameters = [[1.0, 2.0], [1.2, 2.2]]

    log_likelihood_list = [1.0, 0.0]

    return af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model
        )
    )


def test__results_to_file(samples):
    file_results = path.join(text_path, "model.results")

    text_util.results_to_file(
        samples=samples, filename=file_results
    )

    results = open(file_results)

    line = results.readline()

    assert (
            line
            == "Maximum Log Likelihood                                                                    1.00000000\n"
    )

    line = results.readline()

    assert (
            line
            == "Maximum Log Posterior                                                                     1.00000000\n"
    )


def test__search_summary_to_file(model):
    file_search_summary = path.join(text_path, "search.summary")

    parameters = [[1.0, 2.0], [1.2, 2.2]]

    log_likelihood_list = [1.0, 0.0]

    samples = af.m.MockSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=[0.0, 0.0],
            weight_list=log_likelihood_list,
            model=model
        ),
        time=None,
    )

    text_util.search_summary_to_file(samples=samples, log_likelihood_function_time=1.0, filename=file_search_summary)

    results = open(file_search_summary)
    lines = results.readlines()
    assert lines[0] == "Total Samples = 2\n"
    results.close()

    samples = af.m.MockNestSamples(
        model=model,
        sample_list=af.Sample.from_lists(
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list + [2.0],
            log_prior_list=[1.0, 1.0],
            weight_list=log_likelihood_list,
            model=model
        ),
        total_samples=10,
        time=2,
        number_live_points=1,
        log_evidence=1.0,
    )

    text_util.search_summary_to_file(samples=samples, log_likelihood_function_time=1.0, filename=file_search_summary)

    results = open(file_search_summary)
    lines = results.readlines()
    assert lines[0] == "Total Samples = 10\n"
    assert lines[1] == "Total Accepted Samples = 2\n"
    assert lines[2] == "Acceptance Ratio = 0.2\n"
    assert lines[3] == "Time To Run = 2\n"
    assert lines[4] == "Log Likelihood Function Evaluation Time (seconds) = 1.0"
    results.close()
