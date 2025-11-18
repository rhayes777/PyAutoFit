from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from .analysis import Analysis
from ... import SamplesSummary, AbstractPaths, SamplesPDF


class ModelAnalysis(Analysis):
    def __init__(self, analysis: Analysis, model: AbstractPriorModel, use_jax : bool = False):
        """
        Comprises a model and an analysis that can be applied to instances of that model.

        Parameters
        ----------
        analysis
        model
        """
        super().__init__(use_jax=use_jax)

        self.analysis = analysis
        self.model = model

    def __getattr__(self, item):
        if item in ("__getstate__", "__setstate__"):
            raise AttributeError(item)
        return getattr(self.analysis, item)

    def log_likelihood_function(self, instance):
        return self.analysis.log_likelihood_function(instance)

    def make_result(
        self,
        samples_summary: SamplesSummary,
        paths: AbstractPaths,
        samples: Optional[SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ):
        """
        Return the correct type of result by calling the underlying analysis.
        """
        try:
            return self.analysis.make_result(
                samples_summary=samples_summary,
                paths=paths,
                samples=samples,
                search_internal=search_internal,
            )
        except TypeError:
            raise
