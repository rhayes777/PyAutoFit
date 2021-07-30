from autofit.mapper.model import ModelInstance

from autofit.tools.namer import namer
from .abstract import AbstractDeclarativeFactor


class FactorGraphModel(AbstractDeclarativeFactor):
    @property
    def prior_model(self):
        from autofit.mapper.prior_model.collection import CollectionPriorModel
        return CollectionPriorModel({
            factor.name: factor.prior_model
            for factor
            in self.model_factors
        })

    @property
    def optimiser(self):
        raise NotImplemented()

    def __init__(
            self,
            *model_factors: AbstractDeclarativeFactor,
            name=None
    ):
        """
        A collection of factors that describe models, which can be
        used to create a graph and messages.

        If the models have shared priors then the graph has shared variables

        Parameters
        ----------
        model_factors
        """
        self._model_factors = list(model_factors)
        self._name = name or namer(self.__class__.__name__)

    @property
    def name(self):
        return self._name

    def add(self, model_factor: AbstractDeclarativeFactor):
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
        log_likelihood = 0
        for model_factor, instance_ in zip(
                self.model_factors,
                instance
        ):
            log_likelihood += model_factor.log_likelihood_function(
                instance_
            )

        return log_likelihood

    @property
    def model_factors(self):
        return self._model_factors
