import pytest

import autofit as af


def test_trivial():
    instance = af.ModelInstance(items=dict(t=1))
    time_series = af.LinearInterpolator([instance])

    result = time_series[time_series.t == 1]

    assert result is instance


@pytest.fixture(name="time_series")
def make_time_series():
    return af.LinearInterpolator(
        [
            af.ModelInstance(
                items=dict(
                    t=1.0,
                    gaussian=af.Gaussian(centre=0.0, normalization=1.0, sigma=-1.0),
                )
            ),
            af.ModelInstance(
                items=dict(
                    t=2.0,
                    gaussian=af.Gaussian(centre=1.0, normalization=2.0, sigma=-2.0),
                )
            ),
        ]
    )


@pytest.mark.parametrize(
    "t, centre", [(0.0, -1.0), (1.0, 0.0), (1.5, 0.5), (2.0, 1.0), (3.0, 2.0)]
)
def test_linear(t, centre, time_series):

    result = time_series[time_series.t == t]

    assert result.t == t
    assert result.gaussian.centre == centre
    assert result.gaussian.normalization == t
    assert result.gaussian.sigma == -t


@pytest.mark.parametrize("sigma", [-0.5, 0.0, 0.5, 1.0])
def test_alternate_attribute(time_series, sigma):

    result = time_series[time_series.gaussian.sigma == sigma]

    assert result.gaussian.sigma == sigma
    assert result.t == -sigma
    assert result.gaussian.normalization == -sigma
