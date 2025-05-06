import logging
from typing import List, Optional

from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from .analysis import Analysis
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.non_linear.samples import SamplesPDF

logger = logging.getLogger(__name__)


class CombinedResult(Result):
    def __init__(
        self,
        results: List[Result],
        samples: Optional[SamplesPDF] = None,
        samples_summary: Optional[SamplesSummary] = None,
        paths: Optional[AbstractPaths] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[Analysis] = None,
    ):
        """
        A `Result` object that is composed of multiple `Result` objects. This is used to combine the results of
        multiple `Analysis` objects into a single `Result` object, for example when performing a model-fitting
        analysis where there are multiple datasets.

        Parameters
        ----------
        results
            The list of `Result` objects that are combined into this `CombinedResult` object.
        """
        super().__init__(
            samples_summary=samples_summary,
            samples=samples,
            paths=paths,
            search_internal=search_internal,
            analysis=analysis,
        )
        self.child_results = results

    def __getattr__(self, item: str):
        """
        Get an attribute of the first `Result` object in the list of `Result` objects.
        """
        if item in ("__getstate__", "__setstate__"):
            raise AttributeError(item)
        return getattr(self.child_results[0], item)

    def __iter__(self):
        return iter(self.child_results)

    def __len__(self):
        return len(self.child_results)

    def __getitem__(self, item: int) -> Result:
        """
        Get a `Result` object from the list of `Result` objects.
        """
        return self.child_results[item]
