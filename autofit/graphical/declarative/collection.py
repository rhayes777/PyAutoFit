from typing import Union, Optional

from autofit.graphical.declarative.factor.hierarchical import HierarchicalFactor
from autofit.mapper.model import ModelInstance
from autofit.tools.namer import namer
from .abstract import AbstractDeclarativeFactor
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.samples.pdf import SamplesPDF
from autofit.non_linear.samples.summary import SamplesSummary
from autofit.non_linear.analysis.combined import CombinedResult


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
        child_results = [
            model_factor.analysis.make_result(
                samples_summary=samples_summary.subsamples(model_factor.prior_model),
                paths=paths,
                samples=samples,
                search_internal=search_internal,
                analysis=analysis,
            )
            for model_factor in self.model_factors
        ]
        return CombinedResult(
            child_results,
            samples,
            samples_summary,
        )
