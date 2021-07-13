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


def test_query(
        session
):
    fit = db.Fit(
        id="test_query"
    )
    fit.named_instances[
        "one"
    ] = af.Gaussian()

    session.add(fit)
    session.commit()

    fit, = db.Fit.all(
        session
    )

    assert isinstance(
        fit.named_instances[
            "one"
        ],
        af.Gaussian
    )
