from math import prod


def test_run(masked_sensitivity):
    result = masked_sensitivity.run()
    number_elements = prod(masked_sensitivity.shape)
    assert len(result.samples) == number_elements
