from typing import Optional

from autofit.aggregator import Aggregator

import csv


class Column:
    def __init__(
        self,
        column: str,
        name: Optional[str] = None,
    ):
        self.column = column
        self.name = name or column.replace(
            ".",
            "_",
        )


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
                samples_summary = result.value("samples_summary")
                print(samples_summary)
