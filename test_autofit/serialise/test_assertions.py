import itertools

import pytest

import autofit as af
from autofit.mapper.prior.arithmetic.assertion import Compound


@pytest.fixture(name="assertion_dict")
def make_assertion_dict():
    return {
        "type": "compound",
        "compound_type": "GreaterThanLessThanAssertion",
        "left": 0,
        "right": {
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
        "arguments": {
            "centre": {
                "lower_limit": 0.0,
                "upper_limit": 1.0,
                "type": "Uniform",
                "id": 0,
            },
            "normalization": {
                "lower_limit": 0.0,
                "upper_limit": 1.0,
                "type": "Uniform",
                "id": 1,
            },
            "sigma": {
                "lower_limit": 0.0,
                "upper_limit": 1.0,
                "type": "Uniform",
                "id": 2,
            },
        },
        "assertions": [assertion_dict],
    }


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian)


def test_to_dict(model_dict, model):
    model.add_assertion(model.centre > 0)

    assert model.dict() == model_dict


def test_from_dict(assertion_dict):
    assertion = Compound.from_dict(assertion_dict)

    assert assertion.left == 0
    assert isinstance(assertion, af.GreaterThanLessThanAssertion)
    assert isinstance(assertion.right, af.UniformPrior)


def test_model_from_dict(model_dict):
    model = af.Model.from_dict(model_dict)

    (assertion,) = model.assertions
    assert assertion.right is model.centre


def test_compound_assertion(model_dict, model):
    assertion = (model.centre > 0) < 1

    assertion_dict = assertion.dict()
    assertion = Compound.from_dict(assertion_dict)

    assertion_1 = assertion.assertion_1
    assertion_2 = assertion.assertion_2

    assert isinstance(assertion_1, af.GreaterThanLessThanAssertion)
    assert isinstance(assertion_2, af.GreaterThanLessThanAssertion)

    assert assertion_1.left == 0
    assert assertion_2.right == 1
