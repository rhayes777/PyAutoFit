from sqlalchemy import and_, exists

from autofit import database as db
from autofit.mock import mock as m


def test_query(
        session
):
    gaussian_1 = db.Object.from_object(
        m.Gaussian(
            centre=1
        )
    )
    gaussian_2 = db.Object.from_object(
        m.Gaussian(
            centre=2
        )
    )

    session.add_all([
        gaussian_1,
        gaussian_2
    ])
    session.commit()

    result = db.Aggregator(session).filter(
        centre=1
    )

    assert result == [gaussian_1]
