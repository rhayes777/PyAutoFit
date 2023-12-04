import pytest

from autofit.non_linear.grid.sensitivity.job import JobResult
from autofit.non_linear.grid.sensitivity.result import SensitivityResult


class Samples:

    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood

    def summary(self):
        return self

class Result:

    def __init__(self, samples):
        self.samples = samples

    @property
    def log_likelihood(self):
        return self.samples.log_likelihood


@pytest.fixture(name="job_result")
def make_result():
    return JobResult(
        number=0,
        result=Result(Samples(log_likelihood=1.0)),
        perturb_result=Result(Samples(log_likelihood=2.0)),
    )


def test_job_result(job_result):
    assert job_result.log_likelihood_increase == 1.0


def test_result(job_result):
    result = SensitivityResult(
        samples=[job_result.result.samples.summary()],
        perturb_samples=[job_result.perturb_result.samples.summary()],
        shape=(1,)
    )

    assert result.log_likelihoods_base == [1.0]
    assert result.log_likelihoods_perturbed == [2.0]
    assert result.log_likelihood_differences == [1.0]
