from autofit.aggregator import Aggregator


class AggregateSummary:
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def add_column(self, name):
        pass

    def save(self, path):
        with open(path, "w") as f:
            f.write("")
