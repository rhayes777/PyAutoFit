import autofit as af
from autofit import database as db


def test_set_and_retrieve(
        session
):
    fit = db.Fit()
    fit.named_instances[
        "one"
    ] = af.Gaussian()

    assert isinstance(
        fit.named_instances[
            "one"
        ],
        af.Gaussian
    )
