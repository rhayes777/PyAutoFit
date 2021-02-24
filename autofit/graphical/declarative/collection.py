from autofit import ModelInstance
from autofit.mapper.prior_model.collection import CollectionPriorModel

from .abstract import AbstractModelFactor


class FactorGraphModel(AbstractModelFactor):
    @property
    def prior_model(self):
        return CollectionPriorModel(
            factor.prior_model
            for factor
            in self.model_factors
        )

    @property
    def optimiser(self):
        raise NotImplemented()

    def __init__(self, *model_factors: AbstractModelFactor):
        """
        A collection of factors that describe models, which can be
        used to create a graph and messages.

        If the models have shared priors then the graph has shared variables

        Parameters
        ----------
        model_factors
        """
        self._model_factors = list(model_factors)

    def add(self, model_factor: AbstractModelFactor):
        self._model_factors.append(
            model_factor
        )

    def log_likelihood_function(
            self,
            instance: ModelInstance
    ) -> float:
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
        likelihood = abs(
            self.model_factors[0].log_likelihood_function(
                instance[0]
            )
        )
        for model_factor, instance_ in zip(
                self.model_factors[1:],
                instance[1:]
        ):
            likelihood *= abs(
                model_factor.log_likelihood_function(
                    instance_
                )
            )
        return -likelihood

    @property
    def model_factors(self):
        return self._model_factors
