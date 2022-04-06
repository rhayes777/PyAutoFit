import logging
from typing import Tuple

from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import CollectionPriorModel
from .analysis import Analysis
from .combined import CombinedAnalysis

logger = logging.getLogger(
    __name__
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
        child_results = [
            analysis.make_result(
                samples.subsamples(model),
                model,
                search
            )
            for model, analysis in zip(model, self.analyses)
        ]
        result = self.analyses[0].make_result(
            samples=samples,
            model=model,
            search=search,

        )
        result.child_results = child_results
        return result
