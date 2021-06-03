import autofit as af
from autofit import database as db
from autofit.mock.mock import Gaussian


def test_instance_from_prior_medians():
    model = af.Model(
        Gaussian
    )
    db.Object.from_object(
        model
    )()
    db.Object.from_object(
        Gaussian()
    )()
    instance = model.instance_from_prior_medians()
    db.Object.from_object(
        instance
    )()
