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
    Median = 0
    MaxLogLikelihood = 1
    ValuesAt1Sigma = 2
    ValuesAt3Sigma = 3


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
            result[""] = row.median_pdf_sample_kwargs[self.path]

        if ValueType.MaxLogLikelihood in self.value_types:
            result["max_lh"] = row.max_likelihood_kwargs

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
            return self.compute(row.result.samples)
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
