from dataclasses import dataclass

from autofit.aggregator import Aggregator

import csv


@dataclass
class Column:
    argument: str


class AggregateSummary:
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def add_column(self, argument):
        pass

    def save(self, path):
        with open(path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id"],
            )
            writer.writeheader()
            for result in self.aggregator:
                writer.writerow({"id": result.id})
