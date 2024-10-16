from math import prod

import pytest


@pytest.fixture(name="masked_result")
def make_masked_result(masked_sensitivity):
    return masked_sensitivity.run()


def test_run(masked_sensitivity, masked_result):
    number_elements = prod(masked_sensitivity.shape)
    assert len(masked_result.samples) == number_elements
