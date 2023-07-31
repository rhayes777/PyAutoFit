import pytest

import autofit as af


@pytest.fixture(name="assertion_dicts")
def make_assertion_dicts():
    return [
        {
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
    ]


def test_to_dict(assertion_dicts):
    model = af.Model(af.Gaussian)

    model.add_assertion(model.centre > 0)

    assert model.dict()["assertions"] == assertion_dicts
