import autofit as af
from test_autofit import mock


def test_model_with_type(phase, collection):
    promise = phase.result.model[mock.MockComponents][0]

    result = promise.populate(collection)

    assert isinstance(result, af.PriorModel)
    assert result.cls == mock.MockComponents


def test_instance_with_type(phase, collection):
    promise = phase.result.instance[mock.MockComponents][0]

    result = promise.populate(collection)

    assert isinstance(result, mock.MockComponents)
