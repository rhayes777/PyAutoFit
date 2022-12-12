import autofit as af


def test_trivial():
    instance = af.ModelInstance(items=dict(t=1, model=af.Gaussian()),)
    time_series = af.TimeSeries([instance])

    result = time_series[time_series.t == 1]

    assert result is instance
