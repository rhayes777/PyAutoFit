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
        self._analysis_pool = None
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

        def func(child_paths, analysis):
            return analysis.modify_before_fit(child_paths, model)

        return CombinedAnalysis(*self._for_each_analysis(func, paths))

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

        def func(child_paths, analysis, result_):
            return analysis.modify_after_fit(child_paths, model, result_)

        return CombinedAnalysis(
            *self._for_each_analysis(func, paths, result.child_results)
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
            self._analysis_pool = AnalysisPool(self.analyses, self.n_cores)
            self._log_likelihood_function = self._analysis_pool
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

    def _for_each_analysis(self, func, paths, *args) -> List[Union[Result, Analysis]]:
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
        results = []
        for (i, analysis), *args in zip(enumerate(self.analyses), *args):
            child_paths = paths.for_sub_analysis(analysis_name=f"analyses/analysis_{i}")
            results.append(func(child_paths, analysis, *args))

        return results

    def save_attributes(self, paths: AbstractPaths):
        def func(child_paths, analysis):
            analysis.save_attributes(
                child_paths,
            )

        self._for_each_analysis(func, paths)

    def save_results(self, paths: AbstractPaths, result: Result):
        def func(child_paths, analysis, result_):
            analysis.save_results(paths=child_paths, result=result_)

        self._for_each_analysis(func, paths, result)

    def visualize_before_fit(self, paths: AbstractPaths, model: AbstractPriorModel):
        """
        Visualise the model before fitting.

        Visualisation output is distinguished by using an integer suffix
        for each analysis path.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        """
        if self._analysis_pool:
            self._analysis_pool.map(
                "visualize_before_fit",
                paths,
                model,
            )
            return

        def func(child_paths, analysis):
            analysis.visualize_before_fit(child_paths, model)

        self._for_each_analysis(func, paths)

    def visualize_before_fit_combined(
        self, analyses, paths: AbstractPaths, model: AbstractPriorModel
    ):
        """
        Visualise images and quantities which are shared across all analyses.

        For example, each Analysis may have a different dataset, where the data in each dataset is intended to all
        be plotted on the same matplotlib subplot. This function can be overwritten to allow the visualization of such
        a plot.

        Only the first analysis is used to visualize the combined results, where it is assumed that it uses the
        `analyses` property to access the other analyses and perform visualization.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        """
        self.analyses[0].visualize_before_fit_combined(
            analyses=self.analyses,
            paths=paths,
            model=model,
        )

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
        if self._analysis_pool:
            self._analysis_pool.map(
                "visualize",
                paths,
                instance,
                during_analysis,
            )
            return

        def func(child_paths, analysis):
            analysis.visualize(child_paths, instance, during_analysis)

        self._for_each_analysis(func, paths)

    def visualize_combined(
        self,
        analyses: List["Analysis"],
        instance,
        paths: AbstractPaths,
        during_analysis,
    ):
        """
        Visualise the instance using images and quantities which are shared across all analyses.

        For example, each Analysis may have a different dataset, where the fit to each dataset is intended to all
        be plotted on the same matplotlib subplot. This function can be overwritten to allow the visualization of such
        a plot.

        Only the first analysis is used to visualize the combined results, where it is assumed that it uses the
        `analyses` property to access the other analyses and perform visualization.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        instance
            The maximum likelihood instance of the model so far in the non-linear search.
        during_analysis
            Is this visualisation during analysis?
        """
        self.analyses[0].visualize_combined(
            analyses=self.analyses,
            paths=paths,
            instance=instance,
            during_analysis=during_analysis,
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

        self._for_each_analysis(func, paths)

    def make_result(self, samples):
        child_results = [
            analysis.make_result(
                samples,
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
