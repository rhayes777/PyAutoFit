from autofit.graphical.expectation_propagation import EPHistory
from autofit.graphical.expectation_propagation.ep_mean_field import EPMeanField
from autofit.mapper.prior_model.collection import CollectionPriorModel


class EPResult:
    def __init__(
            self,
            ep_history: EPHistory,
            declarative_factor,
            updated_ep_mean_field: EPMeanField,
    ):
        """
        The result of an EP Optimisation including its history, a declarative
        representation of the optimised graph and the resultant EPMeanField
        which comprises factors and resultant approximations of variables.

        Parameters
        ----------
        ep_history
            A history of the optimisation
        declarative_factor: AbstractDeclarativeFactor
            A declarative representation of the factor being optimised
        updated_ep_mean_field
            An updated mean field; effectively the result of the optimisation
        """
        self.ep_history = ep_history
        self.declarative_factor = declarative_factor
        self.updated_ep_mean_field = updated_ep_mean_field

    @property
    def model(self) -> CollectionPriorModel:
        """
        A collection populated with messages representing the posteriors of
        the EP Optimisation. Each item in the collection represents a single
        factor in the optimisation.
        """
        collection = CollectionPriorModel({
            factor.name: factor.prior_model
            for factor
            in self.declarative_factor.model_factors
        })
        arguments = {
            prior: self.updated_ep_mean_field.mean_field[
                prior
            ]
            for prior
            in collection.priors
        }

        return collection.gaussian_prior_model_for_arguments(
            arguments
        )

    @property
    def instance(self):
        """
        The median instance taken from the updated model
        """
        return self.model.instance_from_prior_medians()
