from autofit.mapper.prior_model.collection import CollectionPriorModel


class EPResult:
    def __init__(
            self,
            ep_history,
            declarative_factor,
            updated_ep_mean_field,
    ):
        self.ep_history = ep_history
        self.declarative_factor = declarative_factor
        self.updated_ep_mean_field = updated_ep_mean_field

    @property
    def model(self):
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
