import logging
from abc import ABC
from typing import Tuple, Union

from autoconf import conf
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.analysis.multiprocessing import AnalysisPool
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

    def __radd__(self, other):
        """
        Allows analysis to be used in sum
        """
        if other == 0:
            return self
        return self + other


class CombinedAnalysis(Analysis):
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
                child_paths,
                model,
                samples
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
            The maximum likliehood instance of the model so far in the non-linear search.
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
        return [analysis.make_result(samples, model, search) for analysis in self.analyses]

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
            return CombinedAnalysis(
                *self.analyses,
                *other.analyses
            )
        return CombinedAnalysis(
            *self.analyses,
            other
        )

    def with_free_parameters(
            self,
            *free_parameters: Union[Prior, TuplePrior, AbstractPriorModel]
    ) -> "FreeParameterAnalysis":
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
        return FreeParameterAnalysis(
            *self.analyses,
            free_parameters=free_parameters
        )


class IndexedAnalysis(Analysis):
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
        self.analysis = analysis
        self.index = index

    def log_likelihood_function(self, instance):
        return self.analysis.log_likelihood_function(
            instance[self.index]
        )


class FreeParameterAnalysis(CombinedAnalysis):
    def __init__(
            self,
            *analyses: Analysis,
            free_parameters: Tuple[Prior, ...]
    ):
        """
        A combined analysis with free parameters.

        All parameters for the model are shared across every analysis except
        for the free parameters which are allowed to vary for individual
        analyses.

        Parameters
        ----------
        analyses
            A list of analyses
        free_parameters
            A list of priors which are independent for each analysis
        """
        super().__init__(*[
            IndexedAnalysis(
                analysis,
                index,
            )
            for index, analysis
            in enumerate(analyses)
        ])
        self.free_parameters = [
            parameter for parameter
            in free_parameters
            if isinstance(
                parameter,
                Prior
            )
        ]
        # noinspection PyUnresolvedReferences
        self.free_parameters += [
            prior
            for parameter
            in free_parameters
            if isinstance(
                parameter,
                (AbstractPriorModel, TuplePrior)
            )
            for prior
            in parameter.priors
        ]

    def modify_model(
            self,
            model: AbstractPriorModel
    ) -> AbstractPriorModel:
        """
        Create prior models where free parameters are replaced with new
        priors. Return those prior models as a collection.

        The number of dimensions of the new prior model is the number of the
        old one plus the number of free parameters multiplied by the number
        of free parameters.

        Parameters
        ----------
        model
            The original model

        Returns
        -------
        A new model with all the same priors except for those associated
        with free parameters.
        """
        return CollectionPriorModel([
            model.mapper_from_partial_prior_arguments({
                free_parameter: free_parameter.new()
                for free_parameter in self.free_parameters
            })
            for _ in self.analyses
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
        return [
            analysis.make_result(
                samples.subsamples(model),
                model,
                search
            )
            for model, analysis in zip(model, self.analyses)
        ]
