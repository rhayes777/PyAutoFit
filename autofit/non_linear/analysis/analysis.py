import logging
from abc import ABC

from autoconf import conf
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.analysis.multiprocessing import AnalysisPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from autofit.non_linear.samples import OptimizerSamples

logger = logging.getLogger(
    __name__
)


class Analysis(ABC):
    """
    Protocol for an analysis. Defines methods that can or
    must be implemented to define a class that compute the
    likelihood that some instance fits some data.
    """

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        pass

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        pass

    def save_results_for_aggregator(self, paths: AbstractPaths, model: CollectionPriorModel,
                                    samples: OptimizerSamples):
        pass

    def make_result(self, samples, model, search):
        return Result(samples=samples, model=model, search=search)

    def __add__(
            self,
            other: "Analysis"
    ) -> "CombinedAnalysis":
        """
        Analyses can be added together. The resultant
        log likelihood function returns the sum of the
        underlying log likelihood functions.

        Parameters
        ----------
        other
            Another analysis class

        Returns
        -------
        A class that computes log likelihood based on both analyses
        """
        if isinstance(
                other,
                CombinedAnalysis
        ):
            return other + self
        return CombinedAnalysis(
            self, other
        )


class CombinedAnalysis(Analysis):
    def __init__(self, *analyses: Analysis):
        """
        Computes the summed log likelihood of multiple analyses
        applied to a single model.

        Parameters
        ----------
        analyses
        """
        self.analyses = analyses

        n_cores = conf.instance[
            "general"
        ][
            "analysis"
        ][
            "n_cores"
        ]

        if n_cores > 1:
            self.log_likelihood_function = AnalysisPool(
                analyses,
                n_cores
            )
        else:
            self.log_likelihood_function = lambda instance: sum(
                analysis.log_likelihood_function(
                    instance
                )
                for analysis in analyses
            )

    def __len__(self):
        return len(self.analyses)

    def __add__(self, other):
        if isinstance(
                other,
                CombinedAnalysis
        ):
            return CombinedAnalysis(
                *self.analyses,
                *other.analyses
            )
        return CombinedAnalysis(
            *self.analyses,
            other
        )

    def log_likelihood_function(
            self,
            instance
    ) -> float:
        pass
