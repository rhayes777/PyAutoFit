from typing import Set, Optional

from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from .abstract import AbstractModelFactor


class HierarchicalFactor(AbstractModelFactor):
    def __init__(
            self,
            prior_model: AbstractPriorModel,
            argument_prior: Prior,
            optimiser=None,
            name: Optional[str] = None
    ):
        """
        A factor that links a variable to a parameterised distribution.

        Parameters
        ----------
        prior_model
            A prior model which parameterises a distribution from which it
            is assumed the variable is drawn
        argument_prior
            A prior representing a variable which was drawn from the distribution
        optimiser
            An optional optimiser for optimisation of this factor
        name
            An optional name to distinguish this factor
        """
        self.argument_prior = argument_prior

        def _factor(
                **kwargs
        ):
            argument = kwargs.pop(
                "argument"
            )
            arguments = dict()
            for name_, array in kwargs.items():
                prior_id = int(name_.split("_")[1])
                prior = prior_model.prior_with_id(
                    prior_id
                )
                arguments[prior] = array
            result = prior_model.instance_for_arguments(
                arguments
            )(argument)
            return result

        prior_variable_dict = {
            prior.name: prior
            for prior
            in prior_model.priors
        }

        prior_variable_dict[
            "argument"
        ] = argument_prior

        super().__init__(
            prior_model=prior_model,
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
        to parameterise the distribution, plus an additional prior for the
        variable drawn from the distribution.
        """
        priors = super().priors
        priors.add(
            self.argument_prior
        )
        return priors
