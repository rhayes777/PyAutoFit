import autofit as af


def test_aggregate(aggregator):
    summary = af.AggregateFITS(aggregator)
    summary.extract_fits()
