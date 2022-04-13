import logging
from abc import ABC

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from autofit.non_linear.samples import Samples

logger = logging.getLogger(
    __name__
)


class Analysis(ABC):
    """
    Protocol for an analysis. Defines methods that can or
    must be implemented to define a class that compute the
    likelihood that some instance fits some data.
    """

    def with_model(self, model):
        """
        Associate an explicit model with this analysis. Instances of the model
        will be used to compute log likelihood in place of the model passed
        from the search.

        Parameters
        ----------
        model
            A model to associate with this analysis

        Returns
        -------
        An analysis for that model
        """
        from .model_analysis import ModelAnalysis
        return ModelAnalysis(
            analysis=self,
            model=model
        )

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        pass

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        pass

    def save_results_for_aggregator(self, paths: AbstractPaths, model: CollectionPriorModel,
                                    samples: Samples):
        pass

    def modify_before_fit(self, paths: AbstractPaths, model: AbstractPriorModel):
        """
        Overwrite this method to modify the attributes of the `Analysis` class before the non-linear search begins.

        An example use-case is using properties of the model to alter the `Analysis` class in ways that can speed up
        the fitting performed in the `log_likelihood_function`.
        """
        return self

    def modify_model(self, model):
        return model

    def modify_after_fit(self, paths: AbstractPaths, model: AbstractPriorModel, result: Result):
        """
        Overwrite this method to modify the attributes of the `Analysis` class before the non-linear search begins.

        An example use-case is using properties of the model to alter the `Analysis` class in ways that can speed up
        the fitting performed in the `log_likelihood_function`.
        """
        return self

    def make_result(self, samples, model, search):
        return Result(samples=samples, model=model, search=search)

    def profile_log_likelihood_function(self, paths: AbstractPaths, instance):
        """
        Overwrite this function for profiling of the log likelihood function to be performed every update of a 
        non-linear search.
        
        This behaves analogously to overwriting the `visualize` function of the `Analysis` class, whereby the user 
        fills in the project-specific behaviour of the profiling.
        
        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        instance
            The maximum likliehood instance of the model so far in the non-linear search.
        """
        pass

    def __add__(
            self,
            other: "Analysis"
    ):
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
        from .combined import CombinedAnalysis
        if isinstance(
                other,
                CombinedAnalysis
        ):
            return other + self
        return CombinedAnalysis(
            self, other
        )

    def __radd__(self, other):
        """
        Allows analysis to be used in sum
        """
        if other == 0:
            return self
        return self + other
