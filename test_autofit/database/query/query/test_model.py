import autofit as af
from autofit import database as db


def test_embedded_query(
        session,
        aggregator
):
    model_1 = db.Fit(
        id="model_1",
        instance=af.Collection(
            gaussian=af.Gaussian(
                centre=1
            )
        ),
        info={"info": 3}
    )
    model_2 = db.Fit(
        id="model_2",
        instance=af.Collection(
            gaussian=af.Gaussian(
                centre=2
            )
        ),
        info={"info": 4}
    )

    session.add_all([
        model_1,
        model_2
    ])

    result = aggregator.query(
        aggregator.model.centre == 0
    )

    assert result == []

    result = aggregator.query(
        aggregator.model.gaussian.centre == 1
    )

    assert result == [model_1]

    result = aggregator.query(
        aggregator.model.gaussian.centre == 2
    )

    assert result == [model_2]


def test_query(
        aggregator,
        gaussian_1,
        gaussian_2
):
    result = aggregator.query(
        aggregator.model.centre == 0
    )

    assert result == []

    result = aggregator.query(
        aggregator.model.centre == 1
    )

    assert result == [gaussian_1]

    result = aggregator.query(
        aggregator.model.centre == 2
    )

    assert result == [gaussian_2]
