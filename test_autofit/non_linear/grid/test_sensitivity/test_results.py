from autofit.non_linear.grid.sensitivity import SensitivityResult, JobResult
import pytest


class Result:
    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood


@pytest.fixture(name="job_result")
def make_result():
    return JobResult(
        number=0,
        result=Result(log_likelihood=1.0),
        perturbed_result=Result(log_likelihood=2.0),
    )


def test_job_result(job_result):
    assert job_result.log_likelihood_base == 1.0
    assert job_result.log_likelihood_perturbed == 2.0
    assert job_result.log_likelihood_difference == 1.0


def test_result(job_result):
    result = SensitivityResult(results=[job_result])

    assert result.log_likelihoods_base == [1.0]
    assert result.log_likelihoods_perturbed == [2.0]
    assert result.log_likelihood_differences == [1.0]
