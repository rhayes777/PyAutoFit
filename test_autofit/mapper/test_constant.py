import pytest

import autofit as af
from autofit.mapper.mock.mock_model import WithConstants, Parameter
from autofit.mapper.prior.constant import Constant


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


def test_collection(model):
    collection = af.Collection(model, Constant(1.0))

    instance = collection.instance_from_unit_vector([])
    assert instance[0].constant_1 == 1.0
    assert instance[0].constant_2 == 1.0

    assert instance[1] == 1.0

    assert isinstance(instance[0].constant_1, float)
    assert isinstance(instance[1], float)


def test_kwarg():
    model = af.Model(Parameter, value=1.0)
    assert model.value != af.Model(Parameter, value=1.0).value
