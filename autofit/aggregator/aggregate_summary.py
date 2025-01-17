from typing import Optional

from autofit.aggregator import Aggregator

import csv


class Column:
    def __init__(
        self,
        argument: str,
        name: Optional[str] = None,
    ):
        self.argument = argument
        self.name = name or argument.replace(
            ".",
            "_",
        )

    @property
    def path(self):
        return tuple(self.argument.split("."))


class AggregateSummary:
    def __init__(self, aggregator: Aggregator):
        self._aggregator = aggregator
        self._columns = []

    def add_column(
        self,
        argument: str,
        name: Optional[str] = None,
    ):
        self._columns.append(
            Column(
                argument,
                name=name,
            )
        )

    @property
    def fieldnames(self):
        return ["id"] + [column.name for column in self._columns]

    def save(self, path):
        with open(path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
            )
            writer.writeheader()
            for result in self._aggregator:
                samples_summary = result.value("samples_summary")
                kwargs = samples_summary.median_pdf_sample.kwargs
                row = {"id": result.id}
                for column in self._columns:
                    value = kwargs[column.path]
                    row[column.name] = value

                writer.writerow(row)
