import pytest

import autofit as af


def test_trivial():
    instance = af.ModelInstance(items=dict(t=1))
    time_series = af.LinearTimeSeries([instance])

    result = time_series[time_series.t == 1]

    assert result is instance


@pytest.mark.parametrize(
    "t, centre", [(0.0, -1.0), (1.0, 0.0), (1.5, 0.5), (2.0, 1.0), (3.0, 2.0)]
)
def test_linear(t, centre):
    time_series = af.LinearTimeSeries(
        [
            af.ModelInstance(
                items=dict(
                    t=1, gaussian=af.Gaussian(centre=0.0, normalization=1.0, sigma=-1.0)
                )
            ),
            af.ModelInstance(
                items=dict(
                    t=2, gaussian=af.Gaussian(centre=1.0, normalization=2.0, sigma=-2.0)
                )
            ),
        ]
    )

    result = time_series[time_series.t == t]

    assert result.gaussian.centre == centre
    assert result.gaussian.normalization == t
    assert result.gaussian.sigma == -t
