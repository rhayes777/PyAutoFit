from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable


class AbstractColumn(ABC):
    def __init__(self, name: str):
        """
        A column in the summary table.

        Parameters
        ----------
        name
            An optional name for the column
        """
        self.name = name

    @abstractmethod
    def value(self, row: "Row"):
        pass


class ValueType(Enum):
    """
    Possible value types to create columns for.
    """

    Median = 0  # The median_pdf_sample value
    MaxLogLikelihood = 1  # The max likelihood value
    ValuesAt1Sigma = 2  # The values at 1 sigma. This includes a lower and upper value
    ValuesAt3Sigma = 3  # The values at 3 sigma. This includes a lower and upper value


class Column(AbstractColumn):
    def __init__(
        self,
        argument: str,
        name: Optional[str] = None,
        value_types: list[ValueType] = (ValueType.Median,),
    ):
        """
        A column in the summary table.

        Parameters
        ----------
        argument
            The argument as it appears in the median_pdf_sample arguments
        name
            An optional name for the column
        value_types
            Value types to include. See ValueType.
        """
        super().__init__(
            name
            or argument.replace(
                ".",
                "_",
            )
        )
        self.argument = argument
        self.value_types = value_types

    def value(self, row: "Row"):
        result = {}

        if ValueType.Median in self.value_types:
            try:
                result[""] = row.median_pdf_sample_kwargs[self.path]
            except KeyError:
                result[""] = None

        if ValueType.MaxLogLikelihood in self.value_types:
            try:
                result["max_lh"] = row.max_likelihood_kwargs[self.path]
            except KeyError:
                result["max_lh"] = None

        if ValueType.ValuesAt1Sigma in self.value_types:
            try:
                lower, upper = row.values_at_sigma_1_kwargs[self.path]
                result["lower_1_sigma"] = lower
                result["upper_1_sigma"] = upper
            except KeyError:
                result["lower_1_sigma"] = None
                result["upper_1_sigma"] = None

        if ValueType.ValuesAt3Sigma in self.value_types:
            try:
                lower, upper = row.values_at_sigma_3_kwargs[self.path]
                result["lower_3_sigma"] = lower
                result["upper_3_sigma"] = upper
            except KeyError:
                result["lower_3_sigma"] = None
                result["upper_3_sigma"] = None

        return result

    @property
    def path(self) -> tuple:
        """
        The path of an argument in the median_pdf_sample arguments.
        """
        return tuple(self.argument.split("."))


class ComputedColumn(AbstractColumn):
    def __init__(self, name: str, compute: Callable):
        """
        A column in the summary table that is computed from other columns.

        Parameters
        ----------
        name
            The name of the column
        compute
            A function that takes the median_pdf_sample arguments and returns the computed value
        """
        super().__init__(name)
        self.compute = compute

    def value(self, row: "Row"):

        try:
            return self.compute(row.result)
        except AttributeError as e:
            raise AssertionError(
                "Cannot compute additional fields if no samples.json present"
            ) from e


class LabelColumn(AbstractColumn):
    def __init__(self, name: str, values: list):
        """
        A column in the summary table that is a label.

        Parameters
        ----------
        name
            The name of the column
        """
        super().__init__(name)
        self.values = values

    def value(self, row: "Row"):
        return self.values[row.number]
