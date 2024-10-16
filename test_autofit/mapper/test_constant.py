import pytest

import autofit as af
from autofit.mapper.mock.mock_model import WithConstants


@pytest.fixture
def model():
    return af.Model(WithConstants)


def test_model(model):
    assert model.constant_1 != model.constant_2

    assert model.constant_1 == 1.0
    assert model.constant_2 == 1.0


def test_instance(model):
    instance = model.instance_from_unit_vector([])

    assert instance.constant_1 == 1.0
    assert instance.constant_2 == 1.0

    assert instance.constant_1 == instance.constant_2
