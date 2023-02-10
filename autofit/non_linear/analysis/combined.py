import logging
from typing import Union, List

from autoconf import conf
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.analysis.multiprocessing import AnalysisPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from .analysis import Analysis

logger = logging.getLogger(__name__)


class CombinedResult:
    def __init__(self, results: List[Result]):
        """
        A `Result` object that is composed of multiple `Result` objects. This is used to combine the results of
        multiple `Analysis` objects into a single `Result` object, for example when performing a model-fitting
        analysis where there are multiple datasets.

        Parameters
        ----------
        results
            The list of `Result` objects that are combined into this `CombinedResult` object.
        """
        self.child_results = results

    def __getattr__(self, item: str):
        """
        Get an attribute of the first `Result` object in the list of `Result` objects.
        """
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


class CombinedAnalysis(Analysis):
    def __new__(cls, *analyses, **kwargs):
        from .model_analysis import ModelAnalysis, CombinedModelAnalysis

        if any(isinstance(analysis, ModelAnalysis) for analysis in analyses):
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
        self.n_cores = conf.instance["general"]["analysis"]["n_cores"]

    def __getitem__(self, item):
        return self.analyses[item]

    def modify_before_fit(self, paths: AbstractPaths, model: AbstractPriorModel):
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

    def modify_after_fit(
        self, paths: AbstractPaths, model: AbstractPriorModel, result: Result
    ):
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

    @property
    def n_cores(self):
        return self._n_cores

    @n_cores.setter
    def n_cores(self, n_cores: int):
        """
        Set the number of cores this analysis should use.

        If the number of cores is greater than 1 then log likelihood
        computations are distributed across multiple processes.
        """
        self._n_cores = n_cores
        if self.n_cores > 1:
            analysis_pool = AnalysisPool(self.analyses, self.n_cores)
            self._log_likelihood_function = analysis_pool
        else:
            self._log_likelihood_function = self._summed_log_likelihood

    def _summed_log_likelihood(self, instance) -> float:
        """
        Compute a log likelihood by simply summing the log likelihood
        of each individual analysis computed for some instance.

        Parameters
        ----------
        instance
            An instance of a model

        Returns
        -------
        A combined log likelihood
        """
        return sum(
            analysis.log_likelihood_function(instance) for analysis in self.analyses
        )

    def log_likelihood_function(self, instance):
        return self._log_likelihood_function(instance)

    def _for_each_analysis(self, func, paths, *args):
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
        for (i, analysis), *args in zip(enumerate(self.analyses), *args):
            child_paths = paths.for_sub_analysis(analysis_name=f"analyses/analysis_{i}")
            func(child_paths, analysis, *args)

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        def func(child_paths, analysis):
            analysis.save_attributes_for_aggregator(child_paths,)

        self._for_each_analysis(func, paths)

    def save_results_for_aggregator(self, paths: AbstractPaths, result: Result):
        def func(child_paths, analysis, result_):
            analysis.save_results_for_aggregator(paths=child_paths, result=result_)

        self._for_each_analysis(func, paths, result)

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
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
            analysis.visualize(child_paths, instance, during_analysis)

        self._for_each_analysis(func, paths)

    def profile_log_likelihood_function(
        self, paths: AbstractPaths, instance,
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
                child_paths, instance,
            )

        self._for_each_analysis(func, paths)

    def make_result(self, samples, model, sigma=1.0, use_errors=True, use_widths=False):
        child_results = [
            analysis.make_result(
                samples,
                model,
                sigma=sigma,
                use_errors=use_errors,
                use_widths=use_widths,
            )
            for analysis in self.analyses
        ]
        return CombinedResult(child_results)

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
        if isinstance(other, CombinedAnalysis):
            return type(self)(*self.analyses, *other.analyses)
        return type(self)(*self.analyses, other)

    def with_free_parameters(
        self, *free_parameters: Union[Prior, TuplePrior, AbstractPriorModel]
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

        return FreeParameterAnalysis(*self.analyses, free_parameters=free_parameters)
