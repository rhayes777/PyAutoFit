import pytest

import autofit as af


@pytest.fixture(
    name="sample"
)
def make_sample():
    return af.Sample(
        log_likelihood=0,
        log_prior=0,
        weight=0,
        kwargs={
            ("gaussian_1", "centre",),
            ("gaussian_1", "intensity",),
            ("gaussian_1", "sigma",),
            ("gaussian_2", "centre",),
            ("gaussian_2", "intensity",),
            ("gaussian_2", "sigma",),
        }
    )


def test_trivial(sample):
    with_paths = sample.with_paths([
        ("gaussian_1", "centre",)
    ])

    assert with_paths.kwargs == {
        ("gaussian_1", "centre",)
    }


def test_subpath(sample):
    with_paths = sample.with_paths([
        ("gaussian_1",)
    ])

    assert with_paths.kwargs == {
        ("gaussian_1", "centre",),
        ("gaussian_1", "intensity",),
        ("gaussian_1", "sigma",),
    }
