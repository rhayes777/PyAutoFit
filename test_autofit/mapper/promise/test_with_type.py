import autofit as af
from test_autofit import mock


def test_model_with_type(
        phase,
        collection
):
    promise = phase.result.model[
        mock.Galaxy
    ][0]

    result = promise.populate(collection)

    assert isinstance(result, af.PriorModel)
    assert result.cls == mock.Galaxy
