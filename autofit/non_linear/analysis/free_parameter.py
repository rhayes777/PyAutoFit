import logging
from typing import Tuple

from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import Collection
from .analysis import Analysis
from .indexed import IndexCollectionAnalysis
from ..paths.abstract import AbstractPaths


logger = logging.getLogger(__name__)


class FreeParameterAnalysis(IndexCollectionAnalysis):
    def __init__(self, *analyses: Analysis, free_parameters: Tuple[Prior, ...]):
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
        super().__init__(*analyses)
        self.free_parameters = [
            parameter for parameter in free_parameters if isinstance(parameter, Prior)
        ]
        # noinspection PyUnresolvedReferences
        self.free_parameters += [
            prior
            for parameter in free_parameters
            if isinstance(parameter, (AbstractPriorModel, TuplePrior))
            for prior in parameter.priors
        ]

    def modify_model(self, model: AbstractPriorModel) -> AbstractPriorModel:
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
        return Collection(
            [
                analysis.modify_model(
                    model.mapper_from_partial_prior_arguments(
                        {
                            free_parameter: free_parameter.new()
                            for free_parameter in self.free_parameters
                        }
                    )
                )
                for analysis in self.analyses
            ]
        )

    def modify_before_fit(self, paths: AbstractPaths, model: Collection):
        """
        Modify the analysis before fitting.

        Parameters
        ----------
        paths
            An object describing the paths for saving data (e.g. hard-disk directories or entries in sqlite database).
        model
            The model which is to be fitted.
        """
        return FreeParameterAnalysis(
            *(
                analysis.modify_before_fit(paths, model_)
                for analysis, model_ in zip(self.analyses, model)
            ),
            free_parameters=tuple(self.free_parameters),
        )

    def modify_after_fit(self, paths: AbstractPaths, model: Collection, result):
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
        return FreeParameterAnalysis(
            *(
                analysis.modify_after_fit(paths, model, result)
                for analysis, model_ in zip(self.analyses, model)
            ),
            free_parameters=tuple(self.free_parameters),
        )
