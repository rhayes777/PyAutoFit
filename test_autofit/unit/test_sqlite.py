import autofit as af
from autofit import database as db
from autofit.mock import mock as m


def test_serialise_model():
    model = af.PriorModel(
        m.Gaussian
    )
    result = db.serialize_model(
        model
    )

    assert isinstance(
        result, db.PriorModel
    )
