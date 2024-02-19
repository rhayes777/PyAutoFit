import pytest
import autofit as af
from autofit import SamplesPDF


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.Gaussian)


@pytest.fixture(name="samples")
def make_samples(model):
    return SamplesPDF(
        model=model,
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=3.0,
                kwargs={
                    "centre": 0.0,
                    "normalization": 1.0,
                    "sigma": 1.0,
                },
            ),
        ],
    )
