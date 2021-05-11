from uuid import uuid4

import autofit as af
from autofit.mock.mock import Gaussian


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
                    "gaussian": Gaussian(
                        sigma=sigma
                    )
                }
            )
        )
        session.commit()

    assert len(aggregator) == 3

    assert len(aggregator.query(
        aggregator.gaussian.sigma < 3
    )) == 1
