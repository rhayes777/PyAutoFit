import autofit as af
from autofit.non_linear.grid import sensitivity as s


def test_lists(sensitivity):
    assert len(list(sensitivity._perturbation_instances)) == 8


def test_tuple_step_size(sensitivity):
    sensitivity.number_of_steps = (2, 2, 4)

    assert len(sensitivity._lists) == 16


def test_labels(sensitivity):
    labels = list(sensitivity._labels)
    assert labels == [
        "centre_0.25_normalization_0.25_sigma_0.25",
        "centre_0.25_normalization_0.25_sigma_0.75",
        "centre_0.25_normalization_0.75_sigma_0.25",
        "centre_0.25_normalization_0.75_sigma_0.75",
        "centre_0.75_normalization_0.25_sigma_0.25",
        "centre_0.75_normalization_0.25_sigma_0.75",
        "centre_0.75_normalization_0.75_sigma_0.25",
        "centre_0.75_normalization_0.75_sigma_0.75",
    ]


def test_searches(sensitivity):
    assert len(list(sensitivity._searches)) == 8


def test_perform_job(job):
    result = job.perform()
    assert isinstance(result, s.JobResult)
    assert isinstance(result.perturbed_result, af.Result)
    assert isinstance(result.result, af.Result)


def test_job_paths(
        job,
        search
):
    output_path = search.paths.output_path
    assert job.perturbed_search.paths.output_path == f"{output_path}/[perturbed]"
    assert job.search.paths.output_path == f"{output_path}/[base]"
