from typing import Set, Optional, Type

from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.prior_model import PriorModel
from autofit.messages.abstract import AbstractMessage
from .abstract import AbstractModelFactor


class HierarchicalFactor(PriorModel):
    def __init__(
            self,
            distribution: Type[AbstractMessage],
            **kwargs
    ):
        super().__init__(
            distribution,
            **kwargs
        )
        self.sampled_variables = list()

    def add_sampled_variable(
            self,
            prior: Prior
    ):
        self.sampled_variables.append(
            prior
        )


class _HierarchicalFactor(AbstractModelFactor):
    def __init__(
            self,
            distribution_model: AbstractPriorModel,
            sample_prior: Prior,
            optimiser=None,
            name: Optional[str] = None
    ):
        """
        A factor that links a variable to a parameterised distribution.

        Parameters
        ----------
        distribution_model
            A prior model which parameterizes a distribution from which it
            is assumed the variable is drawn
        sample_prior
            A prior representing a variable which was drawn from the distribution
        optimiser
            An optional optimiser for optimisation of this factor
        name
            An optional name to distinguish this factor
        """
        self.sample_prior = sample_prior

        def _factor(
                **kwargs
        ):
            argument = kwargs.pop(
                "argument"
            )
            arguments = dict()
            for name_, array in kwargs.items():
                prior_id = int(name_.split("_")[1])
                prior = distribution_model.prior_with_id(
                    prior_id
                )
                arguments[prior] = array
            result = distribution_model.instance_for_arguments(
                arguments
            )(argument)
            return result

        prior_variable_dict = {
            prior.name: prior
            for prior
            in distribution_model.priors
        }

        prior_variable_dict[
            "argument"
        ] = sample_prior

        super().__init__(
            prior_model=distribution_model,
            factor=_factor,
            optimiser=optimiser,
            prior_variable_dict=prior_variable_dict,
            name=name
        )

    def log_likelihood_function(self, instance):
        return instance

    @property
    def priors(self) -> Set[Prior]:
        """
        The set of priors associated with this factor. This is the priors used
        to parameterize the distribution, plus an additional prior for the
        variable drawn from the distribution.
        """
        priors = super().priors
        priors.add(
            self.sample_prior
        )
        return priors
