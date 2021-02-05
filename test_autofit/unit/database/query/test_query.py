import autofit as af
from autofit import database as db
from autofit.mock import mock as m


def test_embedded_query(
        session,
        aggregator
):
    model_1 = db.Object.from_object(
        af.Collection(
            gaussian=m.Gaussian(
                centre=1
            )
        )
    )
    model_2 = db.Object.from_object(
        af.Collection(
            gaussian=m.Gaussian(
                centre=2
            )
        )
    )

    session.add_all([
        model_1,
        model_2
    ])

    result = aggregator.filter(
        aggregator.centre == 0
    )

    assert result == []

    result = aggregator.filter(
        aggregator.gaussian.centre == 1
    )

    assert result == [model_1]

    result = aggregator.filter(
        aggregator.gaussian.centre == 2
    )

    assert result == [model_2]


def test_query(
        session,
        aggregator
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

    result = aggregator.filter(
        aggregator.centre == 0
    )

    assert result == []

    result = aggregator.filter(
        aggregator.centre == 1
    )

    assert result == [gaussian_1]

    result = aggregator.filter(
        aggregator.centre == 2
    )

    assert result == [gaussian_2]
