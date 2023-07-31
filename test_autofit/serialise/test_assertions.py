import pytest

import autofit as af
from autofit.mapper.prior.arithmetic.assertion import Assertion


@pytest.fixture(name="assertion_dict")
def make_assertion_dict():
    return {
        "type": "assertion",
        "assertion_type": "GreaterThanLessThanAssertion",
        "lower": 0,
        "greater": {
            "lower_limit": 0.0,
            "upper_limit": 1.0,
            "type": "Uniform",
            "id": 0,
        },
    }


@pytest.fixture(name="model_dict")
def make_model_dict(assertion_dict):
    return {
        "class_path": "autofit.example.model.Gaussian",
        "type": "model",
        "assertions": [assertion_dict],
        "centre": {"lower_limit": 0.0, "upper_limit": 1.0, "type": "Uniform", "id": 0},
        "normalization": {
            "lower_limit": 0.0,
            "upper_limit": 1.0,
            "type": "Uniform",
            "id": 1,
        },
        "sigma": {"lower_limit": 0.0, "upper_limit": 1.0, "type": "Uniform", "id": 2},
    }


def test_to_dict(model_dict):
    model = af.Model(af.Gaussian)

    model.add_assertion(model.centre > 0)

    assert model.dict() == model_dict


def test_from_dict(assertion_dict):
    assertion = Assertion.from_dict(assertion_dict)

    assert assertion.left == 0
    assert isinstance(assertion, af.GreaterThanLessThanAssertion)
    assert isinstance(assertion.right, af.UniformPrior)


def test_model_from_dict(model_dict):
    model = af.Model.from_dict(model_dict)

    (assertion,) = model.assertions
    assert assertion.right is model.centre
