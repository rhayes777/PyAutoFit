import logging

from .analysis import Analysis
from .combined import CombinedAnalysis, CombinedResult
from ..paths.abstract import AbstractPaths

from autofit.mapper.prior_model.collection import Collection

logger = logging.getLogger(__name__)


class IndexedAnalysis:
    def __init__(self, analysis: Analysis, index: int):
        """
        One instance in a collection corresponds to this analysis. That
        instance is identified by its index in the collection.

        Parameters
        ----------
        analysis
            An analysis that can be applied to an instance in a collection
        index
            The index of the instance that should be passed to the analysis
        """
        if isinstance(analysis, IndexedAnalysis):
            analysis = analysis.analysis
        self.analysis = analysis
        self.index = index

    def log_likelihood_function(self, instance):
        """
        Compute the log likelihood by taking the instance at the index
        """
        return self.analysis.log_likelihood_function(instance[self.index])

    # TODO : Add before fit methods here?

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        return self.analysis.visualize(paths, instance[self.index], during_analysis)

    def visualize_combined(
        self, analyses, paths: AbstractPaths, instance, during_analysis
    ):
        return self.analysis.visualize_combined(
            analyses, paths, instance[self.index], during_analysis
        )

    def profile_log_likelihood_function(self, paths: AbstractPaths, instance):
        return self.profile_log_likelihood_function(paths, instance[self.index])

    def __getattr__(self, item):
        if item in ("__getstate__", "__setstate__"):
            raise AttributeError(item)
        return getattr(self.analysis, item)

    def make_result(self, samples):
        return self.analysis.make_result(
            samples,
        )


class IndexCollectionAnalysis(CombinedAnalysis):
    def __init__(self, *analyses):
        """
        Collection of analyses where each analysis has a different
        corresponding model.

        Parameters
        ----------
        analyses
            A list of analyses each with a separate model
        """
        super().__init__(
            *[
                IndexedAnalysis(
                    analysis,
                    index,
                )
                for index, analysis in enumerate(analyses)
            ]
        )

    def make_result(self, samples):
        """
        Associate each model with an analysis when creating the result.
        """
        child_results = [
            analysis.make_result(samples.subsamples(model))
            for model, analysis in zip(samples.model, self.analyses)
        ]
        return CombinedResult(child_results)

    def modify_before_fit(self, paths: AbstractPaths, model: Collection):
        """
        Modify the analysis before fitting.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        model
            The model which is to be fitted.
        """
        return CombinedAnalysis(
            *(analysis.modify_before_fit(paths, model) for analysis in self.analyses)
        )

    def modify_after_fit(self, paths: AbstractPaths, model: Collection, result):
        """
        Modify the analysis after fitting.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        model
            The model which is to be fitted.
        result
            The result of the fit.
        """
        return CombinedAnalysis(
            *(
                analysis.modify_after_fit(paths, model, result)
                for analysis in self.analyses
            )
        )
