from autofit.aggregator import Aggregator


class AggregateFITS:
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def extract_fits(self):
        for result in self.aggregator:
            print(result.fits)
