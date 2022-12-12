import autofit as af


def test_trivial():
    instance = af.ModelInstance(items=dict(t=1))
    time_series = af.TimeSeries([instance])

    result = time_series[time_series.t == 1]

    assert result is instance


def test_linear():
    time_series = af.LinearTimeSeries(
        [
            af.ModelInstance(items=dict(t=1, gaussian=af.Gaussian(centre=0.0))),
            af.ModelInstance(items=dict(t=2, gaussian=af.Gaussian(centre=1.0))),
        ]
    )

    result = time_series[time_series.t == 1.5]

    assert result.gaussian.centre == 0.5
