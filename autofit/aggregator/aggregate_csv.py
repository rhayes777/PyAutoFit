from typing import Optional, Callable, List, Union, Tuple

from pathlib import Path

from autofit.aggregator.aggregator import Aggregator
from autofit.aggregator.search_output import SearchOutput

import csv


class Column:
    def __init__(
        self,
        argument: str,
        name: Optional[str] = None,
        use_max_log_likelihood: Optional[bool] = False,
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
        self.use_max_log_likelihood = use_max_log_likelihood

    @property
    def path(self) -> tuple:
        """
        The path of an argument in the median_pdf_sample arguments.
        """
        return tuple(self.argument.split("."))


class ComputedColumn:
    def __init__(
        self,
        name: str,
        compute: Callable,
    ):
        """
        A column in the summary table that is computed from other columns.

        Parameters
        ----------
        name
            The name of the column
        compute
            A function that takes the median_pdf_sample arguments and returns the computed value
        """
        self.name = name
        self.compute = compute


class Row:
    def __init__(
        self,
        result: SearchOutput,
        columns: List[Column],
        computed_columns: List[ComputedColumn],
    ):
        """
        A row in the summary table corresponding to one search.

        Parameters
        ----------
        result
            The search output from the aggregator
        columns
            The columns to include in the summary. These are taken from samples_summary or latent_summary
        computed_columns
            Columns filled with values computed from the samples
        """
        self._result = result
        self._columns = columns
        self._computed_columns = computed_columns

    def _all_paths(self, path: Tuple[str, ...]):
        model = self._result.model
        return model.all_paths_for_prior(model.object_for_path(path))

    def _add_paths(self, kwargs):
        return {
            path: value
            for key, value in kwargs.items()
            for path in self._all_paths(key)
        }

    @property
    def median_pdf_sample_kwargs(self) -> dict:
        """
        The median_pdf_sample arguments for the search from the samples_summary and latent_summary.
        """
        samples_summary = self._result.value("samples_summary")
        kwargs = self._add_paths(samples_summary.median_pdf_sample.kwargs)

        latent_summary = self._result.value("latent.latent_summary")
        if latent_summary is not None:
            kwargs.update(latent_summary.median_pdf_sample.kwargs)

        return kwargs

    @property
    def max_likelihood_kwargs(self):
        """
        The median_pdf_sample arguments for the search from the samples_summary and latent_summary.
        """
        samples_summary = self._result.value("samples_summary")
        kwargs = self._add_paths(samples_summary.median_pdf_sample.kwargs)

        latent_summary = self._result.value("latent.latent_summary")
        if latent_summary is not None:
            kwargs.update(latent_summary.max_log_likelihood_sample.kwargs)

        return kwargs

    def dict(self) -> dict:
        """
        The row as a dictionary including an id and one entry for each column.
        """
        median_pdf_sample_kwargs = self.median_pdf_sample_kwargs
        max_log_likelihood_sample_kwargs = self.max_likelihood_kwargs

        row = {"id": self._result.id}
        for column in self._columns:
            if column.use_max_log_likelihood:
                kwargs = max_log_likelihood_sample_kwargs
            else:
                kwargs = median_pdf_sample_kwargs

            row[column.name] = kwargs[column.path]

        for column in self._computed_columns:
            try:
                value = column.compute(self._result.samples)
                row[column.name] = value
            except AttributeError as e:
                raise AssertionError(
                    "Cannot compute additional fields if no samples.json present"
                ) from e

        return row


class AggregateCSV:
    def __init__(self, aggregator: Aggregator):
        """
        Summarise results from the aggregator as a CSV.

        Parameters
        ----------
        aggregator
        """
        self._aggregator = aggregator
        self._columns = []
        self._computed_columns = []

    def add_column(
        self,
        argument: str,
        name: Optional[str] = None,
        use_max_log_likelihood: Optional[bool] = False,
    ):
        """
        Add a column to the summary table.

        This will be taken from the samples_summary or latent_summary.

        Parameters
        ----------
        argument
            The argument as it appears in the median_pdf_sample arguments
            e.g. "galaxies.lens.bulge.centre.centre_0"
        name
            An optional name for the column. If not provided, the argument will be used.
        use_max_log_likelihood
            If True, the maximum likelihood value will be used instead of the median PDF value.
        """
        self._columns.append(
            Column(
                argument,
                name=name,
                use_max_log_likelihood=use_max_log_likelihood,
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
        self._computed_columns.append(
            ComputedColumn(
                name,
                compute,
            )
        )

    @property
    def fieldnames(self) -> List[str]:
        """
        The fieldnames for the CSV file.
        """
        return (
            ["id"]
            + [column.name for column in self._columns]
            + [column.name for column in self._computed_columns]
        )

    def save(self, path: Union[str, Path]):
        """
        Save the summary table to a CSV file.

        Parameters
        ----------
        path
            The path to save the file to
        """
        with open(path, "w") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
            )
            writer.writeheader()
            for result in self._aggregator:
                row = Row(
                    result,
                    self._columns,
                    self._computed_columns,
                )

                writer.writerow(row.dict())
