from typing import Optional, Callable, List, Union

from pathlib import Path

from autofit.aggregator.aggregator import Aggregator

import csv

from autofit.aggregator.summary.aggregate_csv.column import (
    LabelColumn,
    Column,
    ComputedColumn,
    ValueType,
)
from autofit.aggregator.summary.aggregate_csv.row import Row


class AggregateCSV:
    def __init__(self, aggregator: Aggregator):
        """
        Summarise results from the aggregator as a CSV.

        Parameters
        ----------
        aggregator
        """
        if len(aggregator) == 0:
            raise ValueError("The aggregator is empty.")

        self._aggregator = aggregator
        self._columns = []

    def add_variable(
        self,
        argument: str,
        name: Optional[str] = None,
        value_types: list = (ValueType.Median,),
    ):
        """
        Add a variable to include columns for in the summary table.

        This will be taken from the samples_summary or latent_summary.

        Parameters
        ----------
        argument
            The argument as it appears in the median_pdf_sample arguments
            e.g. "galaxies.lens.bulge.centre.centre_0"
        name
            An optional name for the column. If not provided, the argument will be used.
        value_types
            Value types to include. See ValueType.
        """
        self._columns.append(
            Column(
                argument,
                name=name,
                value_types=value_types,
            )
        )

    def add_computed_column(
        self,
        name: str,
        compute: Callable,
    ):
        """
        Add a column to the summary table that is computed from other columns.

        Parameters
        ----------
        name
            The name of the column
        compute
            A function that takes the median_pdf_sample arguments and returns the computed value
        """
        self._columns.append(
            ComputedColumn(
                name,
                compute,
            )
        )

    def add_label_column(
        self,
        name: str,
        values: list,
    ):
        """
        Add a column to the summary table that is a label.

        Parameters
        ----------
        name
            The name of the column
        values
            The values of the column
        """
        self._columns.append(
            LabelColumn(
                name,
                values,
            )
        )

    @property
    def fieldnames(self) -> List[str]:
        """
        The fieldnames for the CSV file.
        """
        for i, result in enumerate(self._aggregator):
            row = Row(
                result,
                self._columns,
                i,
            )
            return row.column_names

    def save(self, path: Union[str, Path]):
        """
        Save the summary table to a CSV file.

        Parameters
        ----------
        path
            The path to save the file to
        """

        folder_path = path.parent if path.suffix else path
        folder_path.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
            )
            writer.writeheader()
            for i, result in enumerate(self._aggregator):
                row = Row(
                    result,
                    self._columns,
                    i,
                )

                writer.writerow(row.dict())
