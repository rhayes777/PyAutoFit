from abc import ABC

from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from autofit.non_linear.samples import OptimizerSamples


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

    def log_likelihood_function(
            self,
            instance
    ) -> float:
        """
        Compute the summed log likelihood of all analyses
        applied to the instance

        Parameters
        ----------
        instance
            An instance of a model for some location in
            parameter space

        Returns
        -------
        The summed log likelihood across all contained
        log likelihoods
        """

        def compute_log_likelihood(
                analysis
        ):
            return analysis.log_likelihood_function(
                instance
            )

        return sum(map(
            compute_log_likelihood,
            self.analyses
        ))
