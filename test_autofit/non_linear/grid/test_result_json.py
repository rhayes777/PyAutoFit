import autofit as af
from autofit import Sample

import pytest


@pytest.fixture(name="sample")
def make_sample():
    return Sample(
        log_likelihood=1.0,
        log_prior=2.0,
        weight=3.0,
        kwargs={
            ("centre",): 1.0,
            ("intensity",): 2.0,
            ("sigma",): 3.0,
        },
    )


def test_sample_model_dict(sample):
    assert sample.model_dict() == {
        "centre": 1.0,
        "intensity": 2.0,
        "sigma": 3.0,
    }


def test_embedded_sample_model_dict():
    sample = Sample(
        log_likelihood=1.0,
        log_prior=2.0,
        weight=3.0,
        kwargs={
            (
                "gaussian_1",
                "centre",
            ): 1.0,
            (
                "gaussian_1",
                "intensity",
            ): 2.0,
            (
                "gaussian_1",
                "sigma",
            ): 3.0,
            (
                "gaussian_2",
                "centre",
            ): 1.0,
            (
                "gaussian_2",
                "intensity",
            ): 2.0,
            (
                "gaussian_2",
                "sigma",
            ): 3.0,
        },
    )
    assert sample.model_dict() == {
        "gaussian_1": {
            "centre": 1.0,
            "intensity": 2.0,
            "sigma": 3.0,
        },
        "gaussian_2": {
            "centre": 1.0,
            "intensity": 2.0,
            "sigma": 3.0,
        },
    }


def test_result_json(sample):
    model = af.Model(af.Gaussian)
    result = af.Result(
        samples=af.Samples(
            sample_list=[sample],
            model=model,
        ),
    )

    assert result.dict() == {
        "max_log_likelihood": {
            "centre": 1.0,
            "intensity": 2.0,
            "sigma": 3.0,
        }
    }
