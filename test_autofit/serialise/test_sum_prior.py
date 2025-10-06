import pytest

import autofit as af
from autofit.mapper.prior.arithmetic.compound import SumPrior


@pytest.fixture(name="prior_dict")
def make_prior_dict():
    return {
        "type": "compound",
        "compound_type": "SumPrior",
        "left": {
            "lower_limit": float("-inf"),
            "upper_limit": float("inf"),
            "type": "Gaussian",
            "id": 0,
            "mean": 1.0,
            "sigma": 2.0,
        },
        "right": {
            "lower_limit": float("-inf"),
            "upper_limit": float("inf"),
            "type": "Gaussian",
            "id": 0,
            "mean": 1.0,
            "sigma": 2.0,
        },
    }


def test_sum_prior(prior_dict):
    prior = af.GaussianPrior(
        mean=1.0,
        sigma=2.0,
    )

    sum_prior = prior + prior

    assert sum_prior.dict() == prior_dict


def test_sum_prior_from_dict(prior_dict):
    sum_prior = af.Model.from_dict(prior_dict)

    assert isinstance(sum_prior, SumPrior)
    assert isinstance(sum_prior.left, af.GaussianPrior)
    assert isinstance(sum_prior.right, af.GaussianPrior)
    assert sum_prior.left.mean == 1.0
    assert sum_prior.left.sigma == 2.0
    assert sum_prior.right.mean == 1.0
    assert sum_prior.right.sigma == 2.0
