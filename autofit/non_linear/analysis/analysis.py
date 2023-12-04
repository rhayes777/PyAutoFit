import logging
from abc import ABC
import os

from autoconf import conf

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.database import DatabasePaths
from autofit.non_linear.paths.null import NullPaths
from autofit.non_linear.result import Result

logger = logging.getLogger(__name__)


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

        return ModelAnalysis(analysis=self, model=model)

    def should_visualize(
        self, paths: AbstractPaths, during_analysis: bool = True
    ) -> bool:
        """
        Whether a visualize method should be called perform visualization, which depends on the following:

        1) If a model-fit has already completed, the default behaviour is for visualization to be bypassed in order
        to make model-fits run faster.

        2) If a model-fit has completed, but it is the final visualization output where `during_analysis` is False,
        it should be performed.

        3) Visualization can be forced to run via the `force_visualization_overwrite`, for example if a user
        wants to plot additional images that were not output on the original run.

        4) If the analysis is running a database session visualization is switched off.

        5) If PyAutoFit test mode is on visualization is disabled, irrespective of the `force_visualization_overwite`
        config input.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.


        Returns
        -------
        A bool determining whether visualization should be performed or not.
        """

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return False

        if isinstance(paths, DatabasePaths) or isinstance(paths, NullPaths):
            return False

        if conf.instance["general"]["output"]["force_visualize_overwrite"]:
            return True

        if not during_analysis:
            return True

        return not paths.is_complete

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize_before_fit(self, paths: AbstractPaths, model: AbstractPriorModel):
        pass

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        pass

    def visualize_before_fit_combined(
        self, analyses, paths: AbstractPaths, model: AbstractPriorModel
    ):
        pass

    def visualize_combined(
        self, analyses, paths: AbstractPaths, instance, during_analysis
    ):
        pass

    def save_attributes(self, paths: AbstractPaths):
        pass

    def save_results(self, paths: AbstractPaths, result: Result):
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

    def modify_after_fit(
        self, paths: AbstractPaths, model: AbstractPriorModel, result: Result
    ):
        """
        Overwrite this method to modify the attributes of the `Analysis` class before the non-linear search begins.

        An example use-case is using properties of the model to alter the `Analysis` class in ways that can speed up
        the fitting performed in the `log_likelihood_function`.
        """
        return self

    def make_result(self, samples):
        return Result(
            samples=samples,
        )

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

    def __add__(self, other: "Analysis"):
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

        if isinstance(other, CombinedAnalysis):
            return other + self
        return CombinedAnalysis(self, other)

    def __radd__(self, other):
        """
        Allows analysis to be used in sum
        """
        if other == 0:
            return self
        return self + other
