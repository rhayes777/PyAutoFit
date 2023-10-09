import autofit as af
from autofit.database import Object
from autofit.non_linear.samples.efficient import EfficientSamples


import pytest


@pytest.fixture(name="samples")
def make_samples():
    return af.SamplesPDF(
        model=af.Model(af.Gaussian),
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=4.0,
                kwargs={"centre": 1.0, "normalization": 2.0, "sigma": 3.0},
            )
        ],
    )


@pytest.fixture(name="efficient")
def make_efficient(samples):
    return EfficientSamples(samples=samples)


def test_conversion(efficient):
    samples = efficient.samples

    sample = samples.sample_list[0]
    assert sample.log_likelihood == 1.0
    assert sample.log_prior == 2.0
    assert sample.weight == 4.0


def test_database(efficient, session):
    recovered = Object.from_object(efficient)()
    samples = recovered.samples

    assert samples.model.cls is af.Gaussian
