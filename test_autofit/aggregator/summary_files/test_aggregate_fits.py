import autofit as af


def test_aggregate(aggregator):
    summary = af.AggregateFITS(aggregator)
    result = summary.extract_fits(
        af.FitFITS.ModelImage,
        af.FitFITS.ResidualMap,
    )
    assert len(result) == 5
