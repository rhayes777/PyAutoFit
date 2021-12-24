from uuid import uuid4
import numpy as np

import autofit as af


def test_float_inequality(session):
    aggregator = af.Aggregator(session)

    for sigma in [
        0.9992237362814176,
        4.9687212446221904,
        9.967065800134504,
    ]:
        session.add(
            af.db.Fit(
                id=str(uuid4()),
                instance={
                    "gaussian": af.Gaussian(
                        sigma=sigma
                    )
                }
            )
        )
        session.commit()

    assert len(aggregator) == 3

    assert len(aggregator.query(
        aggregator.model.gaussian.sigma < 3
    )) == 1


def test_numpy_values(
        session
):
    aggregator = af.Aggregator(session)
    fit = af.db.Fit(
        id=str(uuid4())
    )
    array = np.zeros(
        (10, 10)
    )
    fit["data"] = array
    session.add(
        fit
    )
    session.commit()

    assert (
            aggregator.values(
                "data"
            )[0] == array
    ).all()
