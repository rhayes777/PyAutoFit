import numpy as np

import autofit as af
import pytest

from autofit.non_linear.samples.summary import SamplesSummary
from autoconf.dictable import from_dict, to_dict


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


@pytest.fixture(name="summary")
def make_summary(samples_pdf):
    return samples_pdf.summary()


def test_summary(summary, model, sample):
    assert summary.model is model
    assert summary.max_log_likelihood_sample == sample
    assert isinstance(summary.covariance_matrix, np.ndarray)


@pytest.fixture(name="summary_dict")
def make_summary_dict():
    return {
        "arguments": {
            "covariance_matrix": {
                "array": [
                    [2.0, 3.0, 3.9999999999999996],
                    [3.0, 4.5, 6.0],
                    [4.0, 6.0, 7.999999999999999],
                ],
                "dtype": "float64",
                "type": "ndarray",
            },
            "log_evidence": None,
            "max_log_likelihood_sample": {
                "arguments": {
                    "kwargs": {
                        "arguments": {
                            "centre": 2.0,
                            "normalization": 4.0,
                            "sigma": 6.0,
                        },
                        "type": "dict",
                    },
                    "log_likelihood": 4.0,
                    "log_prior": 5.0,
                    "weight": 6.0,
                },
                "class_path": "autofit.non_linear.samples.sample.Sample",
                "type": "instance",
            },
            "model": {
                "arguments": {
                    "centre": {
                        "lower_limit": 0.0,
                        "type": "Uniform",
                        "upper_limit": 1.0,
                    },
                    "normalization": {
                        "lower_limit": 0.0,
                        "type": "Uniform",
                        "upper_limit": 1.0,
                    },
                    "sigma": {
                        "lower_limit": 0.0,
                        "type": "Uniform",
                        "upper_limit": 1.0,
                    },
                },
                "class_path": "autofit.example.model.Gaussian",
                "type": "model",
            },
        },
        "class_path": "autofit.non_linear.samples.summary.SamplesSummary",
        "type": "instance",
    }


def test_dict(summary, summary_dict, remove_ids):
    assert remove_ids(to_dict(summary)) == summary_dict


def test_from_dict(summary_dict):
    summary = from_dict(summary_dict)
    assert isinstance(summary, SamplesSummary)
    assert isinstance(summary.model, af.Model)
    assert isinstance(summary.max_log_likelihood_sample, af.Sample)
    assert isinstance(summary.covariance_matrix, np.ndarray)

    assert isinstance(summary.max_log_likelihood(), af.Gaussian)


def test_generic_from_dict(summary_dict):
    summary = from_dict(summary_dict)
    assert isinstance(summary, SamplesSummary)
    assert isinstance(summary.model, af.Model)
    assert isinstance(summary.max_log_likelihood_sample, af.Sample)
    assert isinstance(summary.covariance_matrix, np.ndarray)


def test_covariance_interpolator(summary):
    interpolator = af.CovarianceInterpolator([summary])
    assert interpolator[interpolator.centre == 0.5]
