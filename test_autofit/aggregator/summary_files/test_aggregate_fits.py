import autofit as af
from autofit.aggregator.summary.aggregate_fits import Fit


def test_aggregate(aggregator):
    summary = af.AggregateFITS(aggregator)
    result = summary.extract_fits(
        Fit.ModelImage,
        Fit.ResidualMap,
    )
    assert len(result) == 5
