from math import prod

import pytest


@pytest.fixture(name="masked_result")
def make_masked_result(masked_sensitivity):
    return masked_sensitivity.run()


def test_result_size(masked_sensitivity, masked_result):
    number_elements = prod(masked_sensitivity.shape)
    assert len(masked_result.samples) == number_elements


def test_sample(masked_result):
    sample = masked_result.samples[0]
    assert sample.model is not None
    assert sample.model.perturb is not None
    assert sample.log_evidence == 0.0
    assert sample.log_likelihood == 0.0
