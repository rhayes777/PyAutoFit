import autofit as af
from autofit import database as db


def test_instance_from_prior_medians():
    model = af.Model(
        af.Gaussian
    )
    db.Object.from_object(
        model
    )()
    db.Object.from_object(
        af.Gaussian()
    )()
    instance = model.instance_from_prior_medians()
    db.Object.from_object(
        instance
    )()
