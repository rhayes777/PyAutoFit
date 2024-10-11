import logging
from collections import defaultdict
from typing import Tuple

from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior.tuple_prior import TuplePrior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import Collection
from .analysis import Analysis
from .indexed import IndexCollectionAnalysis
from ..paths.abstract import AbstractPaths


logger = logging.getLogger(__name__)


def _unpack(free_parameters: Tuple[Prior, ...]):
    """
    Unpack free parameters from a tuple of free parameters and priors.

    Parameters
    ----------
    free_parameters
        A tuple of free parameters and priors.

    Returns
    -------
    A list of free parameters.
    """
    return [
        parameter for parameter in free_parameters if isinstance(parameter, Prior)
    ] + [
        prior
        for parameter in free_parameters
        if isinstance(parameter, (AbstractPriorModel, TuplePrior))
        for prior in parameter.priors
    ]


class PositionalParameters:
    def __init__(
        self,
        analysis: "FreeParameterAnalysis",
        position: int,
    ):
        """
        Manage overriding positional parameters for a given analysis.

        Parameters
        ----------
        analysis
            The analysis
        position
            The position of the model in the collection
        """
        self.analysis = analysis
        self.position = position

    def __setitem__(self, key, value):
        """
        Override some component of the model at the index.

        Parameters
        ----------
        key
            The key of the component (e.g. a prior)
        value
            The new value
        """
        self.analysis.positional_parameters[self.position][key] = value


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
        self.free_parameters = _unpack(free_parameters)
        self.positional_parameters = defaultdict(dict)

    def __getitem__(self, item: int):
        """
        Used to override some model component for the model at a given index.

        Parameters
        ----------
        item
            The index of the model in the collection

        Returns
        -------
        A manager for overriding the model components
        """
        return PositionalParameters(self, item)

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
        collection = Collection(
            [
                analysis.modify_model(
                    model.mapper_from_partial_prior_arguments(self._arguments)
                )
                for analysis in self.analyses
            ]
        )
        for i, positional_parameters in self.positional_parameters.items():
            for key, value in positional_parameters.items():
                path = model.path_for_object(key)
                if path == ():
                    collection[i] = value
                else:
                    collection[i].set_item_at_path(path, value)

        return collection

    @property
    def _arguments(self):
        return {
            free_parameter: free_parameter.new()
            for free_parameter in self.free_parameters
        }

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
