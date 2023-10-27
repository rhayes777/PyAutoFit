from typing import List, Union

from autofit.non_linear.samples import Samples
from autofit.database import Prior
from autofit.non_linear.result import Result, Placeholder
from .job import JobResult
from .result import GridSearchResult


class ResultBuilder:
    def __init__(self, lists: List[List[float]], grid_priors: List[Prior]):
        """
        Builds GridSearchResults including all results so far computed
        and Placeholders where no result has yet been computed.

        Parameters
        ----------
        lists
            A list of lists of unit hypercube vectors describing the
            values that change during the grid search
        grid_priors
            Priors which are varied throughout the grid search
        """
        self.lists = lists
        self.grid_priors = grid_priors
        self._job_result_dict = {}

    def __call__(self) -> GridSearchResult:
        """
        Generate a GridSearchResult with all results so far and placeholders
        where no result has been returned yet
        """
        return GridSearchResult(self.sample_summaries, self.lists, self.grid_priors)

    @property
    def sample_summaries(self) -> List[Union[Samples, Placeholder]]:
        """
        A list of results that have been returned with placeholders where no
        result has been returned in the grid-search order.
        """
        return [samples.summary() for samples in self.samples]

    @property
    def results(self) -> List[Union[Result, Placeholder]]:
        """
        A list of results that have been returned with placeholders where no
        result has been returned in the grid-search order.
        """
        return [
            samples
            if isinstance(samples, Placeholder)
            else Result(
                samples=samples,
            )
            for samples in self.samples
        ]

    @property
    def samples(self) -> List[Union[Samples, Placeholder]]:
        """
        A list of results that have been returned with placeholders where no
        result has been returned in the grid-search order.
        """
        samples = []
        for number in range(len(self.lists)):
            try:
                job_result = self._job_result_dict[number]
                samples.append(
                    job_result.result.samples,
                )
            except KeyError:
                samples.append(Placeholder())
        return samples

    def add(self, job_result: JobResult):
        """
        Add a new result for a completed job.

        The number associated with the job result is used for
        tracking and ordering.

        Parameters
        ----------
        job_result
            A result of an optimisation for a single point in
            the grid.
        """
        self._job_result_dict[job_result.number] = job_result
