import numpy as np

import autofit as af
import pytest


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.Gaussian)


@pytest.fixture(name="sample")
def make_sample(model):
    return af.Sample(
        log_likelihood=4.0,
        log_prior=5.0,
        weight=6.0,
        kwargs={"centre": 2.0, "normalization": 4.0, "sigma": 6.0},
    )


@pytest.fixture(name="samples_pdf")
def make_samples_pdf(model, sample):
    return af.SamplesPDF(
        sample_list=[
            af.Sample(
                log_likelihood=1.0,
                log_prior=2.0,
                weight=3.0,
                kwargs={"centre": 0.0, "normalization": 1.0, "sigma": 2.0},
            ),
            sample,
        ],
        model=model,
    )


def test_summary(samples_pdf, model, sample):
    summary = samples_pdf.summary()

    assert summary.model is model
    assert summary.max_log_likelihood_sample == sample
    assert isinstance(summary.covariance_matrix, np.ndarray)
