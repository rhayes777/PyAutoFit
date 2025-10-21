from typing import Union, Optional

from autofit.graphical.declarative.factor.hierarchical import HierarchicalFactor

from autofit.tools.namer import namer
from .abstract import AbstractDeclarativeFactor
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.non_linear.samples.summary import SamplesSummary

from autofit.mapper.model import ModelInstance
from autofit.mapper.prior_model.prior_model import Model

from autofit.jax_wrapper import register_pytree_node_class
from ...non_linear.combined_result import CombinedResult


@register_pytree_node_class
class FactorGraphModel(AbstractDeclarativeFactor):
    def __init__(
        self,
        *model_factors: Union[AbstractDeclarativeFactor, HierarchicalFactor],
        name=None,
        include_prior_factors=True,
    ):
        """
        A collection of factors that describe models, which can be
        used to create a graph and messages.

        If the models have shared priors then the graph has shared variables

        Parameters
        ----------
        model_factors
            Factors which are hierarchical or associated with a specific analysis
        """
        super().__init__(
            include_prior_factors=include_prior_factors,
        )
        self._model_factors = list(model_factors)
        self._name = name or namer(self.__class__.__name__)

    def tree_flatten(self):
        return (
            (self._model_factors,),
            (self._name, self.include_prior_factors),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            *children[0],
            name=aux_data[0],
            include_prior_factors=aux_data[1],
        )

    @property
    def prior_model(self):
        """
        Construct a Collection comprising the prior models described
        in each model factor
        """
        from autofit.mapper.prior_model.collection import Collection

        return Collection(
            {factor.name: factor.prior_model for factor in self.model_factors}
        )

    @property
    def optimiser(self):
        raise NotImplemented()

    @property
    def info(self) -> str:
        """
        Info describing this collection.
        """
        return self.graph.info

    @property
    def name(self):
        return self._name

    def add(self, model_factor: AbstractDeclarativeFactor):
        """
        Add another factor to this collection.
        """
        self._model_factors.append(model_factor)

    def log_likelihood_function(self, instance: ModelInstance) -> float:
        """
        Compute the combined likelihood of each factor from a collection of instances
        with the same ordering as the factors.

        Parameters
        ----------
        instance
            A collection of instances, one corresponding to each factor

        Returns
        -------
        The combined likelihood of all factors
        """
        log_likelihood = 0
        for model_factor, instance_ in zip(self.model_factors, instance):
            log_likelihood += model_factor.log_likelihood_function(instance_)

        return log_likelihood

    @property
    def model_factors(self):
        model_factors = list()
        for model_factor in self._model_factors:
            if isinstance(model_factor, HierarchicalFactor):
                model_factors.extend(model_factor.factors)
            else:
                model_factors.append(model_factor)
        return model_factors

    def make_result(
        self,
        samples_summary: SamplesSummary,
        paths: AbstractPaths,
        samples: Optional[SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[object] = None,
    ) -> CombinedResult:
        """
        Make a result from the samples summary and paths.

        The top level result accounts for the combined model.
        There is one child result for each model factor.

        Parameters
        ----------
        samples_summary
            A summary of the samples
        paths
            Handles saving and loading data
        samples
            The full list of samples
        search_internal
        analysis

        Returns
        -------
        A result with child results for each model factor
        """
        child_results = [
            model_factor.analysis.make_result(
                samples_summary=samples_summary.subsamples(
                    model_factor.prior_model,
                ),
                paths=paths,
                samples=samples.subsamples(model_factor.prior_model)
                if samples
                else None,
                search_internal=search_internal,
                analysis=model_factor,
            )
            for model_factor in self.model_factors
        ]
        return CombinedResult(
            child_results,
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=analysis,
        )

    def _for_each_analysis(
        self,
        name,
        paths,
        *args,
        **kwargs,
    ):
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
        for (i, analysis), *args in zip(
            enumerate(self.model_factors),
            *args,
        ):
            child_paths = paths.for_sub_analysis(analysis_name=f"analyses/analysis_{i}")
            func = getattr(analysis, name)
            results.append(
                func(
                    child_paths,
                    *args,
                    **kwargs,
                )
            )

        return results

    def visualize(
        self,
        paths: AbstractPaths,
        instance: ModelInstance,
        during_analysis: bool,
    ):
        """
        Visualise the instances provided using each factor.

        Instances in the ModelInstance must have the same order as the factors.

        Parameters
        ----------
        paths
            Object describing where data should be saved to
        instance
            A collection of instances, each corresponding to a factor
        during_analysis
            Is this visualisation during analysis?
        """
        self._for_each_analysis(
            "visualize",
            paths,
            instance,
            during_analysis=during_analysis,
        )

    def visualize_before_fit(
        self,
        paths: AbstractPaths,
        model: Model,
    ):
        """
        Visualise the model provided using each factor.

        Models in the ModelInstance must have the same order as the factors.

        Parameters
        ----------
        paths
            Object describing where data should be saved to
        model
            A collection of models, each corresponding to a factor
        """
        self._for_each_analysis(
            "visualize_before_fit",
            paths,
            model,
        )

    def save_attributes(self, paths: AbstractPaths):
        """
        Save the attributes of the analysis to the paths object.
        """
        self._for_each_analysis("save_attributes", paths)

    def save_results(self, paths: AbstractPaths, result):
        """
        Save the results of the analysis to the paths object.
        """
        self._for_each_analysis("save_results", paths, result)

    def visualize_combined(
        self,
        instance,
        paths: AbstractPaths,
        during_analysis,
    ):
        self.model_factors[0].visualize_combined(
            self.model_factors,
            paths,
            instance,
            during_analysis=during_analysis,
        )

    def perform_quick_update(self, paths, instance):

        try:
            self.model_factors[0].visualize_combined(
                analyses=self.model_factors,
                paths=paths,
                instance=instance,
                during_analysis=True,
                quick_update=True,
            )
        except Exception as e:
            pass