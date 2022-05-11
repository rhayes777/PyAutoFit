import logging
from typing import Union

from autoconf import conf
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.analysis.multiprocessing import AnalysisPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples import Samples
from .analysis import Analysis

logger = logging.getLogger(
    __name__
)


class CombinedAnalysis(Analysis):
    def __new__(cls, *analyses, **kwargs):
        from .model_analysis import ModelAnalysis, CombinedModelAnalysis
        if any(
                isinstance(analysis, ModelAnalysis)
                for analysis in analyses
        ):
            return object.__new__(CombinedModelAnalysis)
        return object.__new__(cls)

    def __init__(self, *analyses: Analysis):
        """
        Computes the summed log likelihood of multiple analyses
        applied to a single model.

        Either analyses are performed sequentially and summed,
        or they are mapped out to processes.

        If the number of cores is greater than one then the
        analyses are distributed across a number of processes
        equal to the number of cores.

        Parameters
        ----------
        analyses
        """
        self.analyses = analyses
        self._n_cores = None
        self._log_likelihood_function = None
        self.n_cores = conf.instance[
            "general"
        ][
            "analysis"
        ][
            "n_cores"
        ]

    @property
    def n_cores(self):
        return self._n_cores

    @n_cores.setter
    def n_cores(self, n_cores):
        self._n_cores = n_cores
        if self.n_cores > 1:
            analysis_pool = AnalysisPool(
                self.analyses,
                self.n_cores
            )
            self._log_likelihood_function = analysis_pool
        else:
            self._log_likelihood_function = lambda instance: sum(
                analysis.log_likelihood_function(
                    instance
                )
                for analysis in self.analyses
            )

    def log_likelihood_function(self, instance):
        return self._log_likelihood_function(instance)

    def _for_each_analysis(self, func, paths):
        """
        Convenience function to call an underlying function for each
        analysis with a paths object with an integer attached to the
        end.

        Parameters
        ----------
        func
            Some function of the analysis class
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        """
        for i, analysis in enumerate(self.analyses):
            child_paths = paths.for_sub_analysis(
                analysis_name=f"analyses/analysis_{i}"
            )
            func(child_paths, analysis)

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        def func(child_paths, analysis):
            analysis.save_attributes_for_aggregator(
                child_paths,
            )

        self._for_each_analysis(
            func,
            paths
        )

    def save_results_for_aggregator(
            self,
            paths: AbstractPaths,
            model: CollectionPriorModel,
            samples: Samples
    ):
        def func(child_paths, analysis):
            analysis.save_results_for_aggregator(
                paths=child_paths,
                model=model,
                samples=samples
            )

        self._for_each_analysis(
            func,
            paths
        )

    def visualize(
            self,
            paths: AbstractPaths,
            instance,
            during_analysis
    ):
        """
        Visualise the instance according to each analysis.

        Visualisation output is distinguished by using an integer suffix
        for each analysis path.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        instance
            The maximum likelihood instance of the model so far in the non-linear search.
        during_analysis
            Is this visualisation during analysis?
        """

        def func(child_paths, analysis):
            analysis.visualize(
                child_paths,
                instance,
                during_analysis
            )

        self._for_each_analysis(
            func,
            paths
        )

    def profile_log_likelihood_function(
            self,
            paths: AbstractPaths,
            instance,
    ):
        """
        Profile the log likelihood function of the maximum likelihood model instance using each analysis.

        Profiling output is distinguished by using an integer suffix for each analysis path.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        instance
            The maximum likliehood instance of the model so far in the non-linear search.
        """

        def func(child_paths, analysis):
            analysis.profile_log_likelihood_function(
                child_paths,
                instance,
            )

        self._for_each_analysis(
            func,
            paths
        )

    def make_result(
            self, samples, model, search
    ):
        child_results = [
            analysis.make_result(
                samples,
                model,
                search
            ) for analysis in self.analyses
        ]
        result = self.analyses[0].make_result(
            samples=samples,
            model=model,
            search=search,

        )
        result.child_results = child_results
        return result

    def __len__(self):
        return len(self.analyses)

    def __add__(self, other: Analysis):
        """
        Adding anything to a CombinedAnalysis results in another
        analysis containing all underlying analyses (no combined
        analysis children)

        Parameters
        ----------
        other
            Some analysis

        Returns
        -------
        An overarching analysis
        """
        if isinstance(
                other,
                CombinedAnalysis
        ):
            return type(self)(
                *self.analyses,
                *other.analyses
            )
        return type(self)(
            *self.analyses,
            other
        )

    def with_free_parameters(
            self,
            *free_parameters: Union[Prior, TuplePrior, AbstractPriorModel]
    ):
        """
        Set some parameters as free parameters. The are priors which vary
        independently for each analysis in the collection.

        Parameters
        ----------
        free_parameters
            Parameters that are allowed to vary independently.

        Returns
        -------
        An analysis with freely varying parameters.
        """
        from .free_parameter import FreeParameterAnalysis
        return FreeParameterAnalysis(
            *self.analyses,
            free_parameters=free_parameters
        )
