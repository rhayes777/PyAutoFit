from math import prod

import pytest

import autofit as af


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


@pytest.mark.parametrize(
    "lower, upper, mean",
    [
        (0.0, 1.0, 0.5),
        (-1.0, 1.0, 0.0),
        (-1.0, 0.0, -0.5),
        (0.5, 1.0, 0.75),
    ],
)
def test_mean_uniform_prior(
    lower,
    upper,
    mean,
):
    prior = af.UniformPrior(
        lower_limit=0.0,
        upper_limit=1.0,
    )
    assert (
        prior.with_limits(
            lower,
            upper,
        ).mean
        == mean
    )


def test_physical_centres_lists(masked_result, masked_sensitivity):
    assert masked_result.perturbed_physical_centres_list_from("perturb.centre") == [
        0.25,
        0.25,
        0.25,
        0.25,
        0.75,
        0.75,
        0.75,
        0.75,
    ]
