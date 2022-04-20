import logging

from .analysis import Analysis
from .combined import CombinedAnalysis
from ..paths.abstract import AbstractPaths

logger = logging.getLogger(
    __name__
)


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
        return self.analysis.log_likelihood_function(
            instance[self.index]
        )

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        return self.analysis.visualize(
            paths, instance[self.index], during_analysis
        )

    def profile_log_likelihood_function(self, paths: AbstractPaths, instance):
        return self.profile_log_likelihood_function(
            paths, instance[self.index]
        )

    def __getattr__(self, item):
        return getattr(
            self.analysis,
            item
        )

    def make_result(self, samples, model, search):
        return self.analysis.make_result(samples, model, search)


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
        super().__init__(*[
            IndexedAnalysis(
                analysis,
                index,
            )
            for index, analysis
            in enumerate(analyses)
        ])

    def make_result(
            self,
            samples,
            model,
            search
    ):
        """
        Associate each model with an analysis when creating the result.
        """
        child_results = [
            analysis.make_result(
                samples.subsamples(model),
                model,
                search
            )
            for model, analysis in zip(model, self.analyses)
        ]
        result = self.analyses[0].make_result(
            samples=samples,
            model=model,
            search=search,

        )
        result.child_results = child_results
        return result
