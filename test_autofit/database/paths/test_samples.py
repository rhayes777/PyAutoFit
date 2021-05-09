import pytest

import autofit as af
from autofit import database as m
from autofit.mock import mock
from autofit.non_linear.samples import Sample


@pytest.fixture(
    name="save_samples",
    autouse=True
)
def save_samples(
        paths,
        sample
):
    samples = mock.MockSamples(
        model=af.Model(
            mock.Gaussian
        ),

    )
    samples._samples = [sample]
    paths.save_samples(
        samples
    )


@pytest.fixture(
    name="sample"
)
def make_sample():
    return Sample(
        log_likelihood=1.0,
        log_prior=1.0,
        weight=0.5,
        centre=1.0,
        intensity=2.0,
        sigma=3.0
    )


def test_serialise_sample(sample):
    assert isinstance(
        sample.kwargs,
        dict
    )

    sample = m.Object.from_object(
        sample
    )()
    assert "centre" in sample.kwargs


def test_load_samples(
        paths
):
    samples = paths._load_samples()

    assert samples.model.cls is mock.Gaussian

    sample, = samples.samples
    assert sample.weight == 0.5
