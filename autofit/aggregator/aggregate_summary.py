from typing import Optional

from autofit.aggregator import Aggregator

import csv


class Column:
    def __init__(
        self,
        argument: str,
        name: Optional[str] = None,
    ):
        """
        A column in the summary table.

        Parameters
        ----------
        argument
            The argument as it appears in the median_pdf_sample arguments
        name
            An optional name for the column
        """
        self.argument = argument
        self.name = name or argument.replace(
            ".",
            "_",
        )

    @property
    def path(self) -> tuple:
        """
        The path of an argument in the median_pdf_sample arguments.
        """
        return tuple(self.argument.split("."))


class AggregateSummary:
    def __init__(self, aggregator: Aggregator):
        """
        Summarise results from the aggregator as a CSV.

        Parameters
        ----------
        aggregator
        """
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

                latent_summary = result.value("latent_summary")
                if latent_summary is not None:
                    kwargs.update(latent_summary.median_pdf_sample.kwargs)

                row = {"id": result.id}
                for column in self._columns:
                    value = kwargs[column.path]
                    row[column.name] = value

                writer.writerow(row)
