from autofit.graphical.factor_graphs.factor import FactorKW
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.analysis import Analysis
from autofit.tools.namer import namer


class PriorFactor(FactorKW, Analysis):
    def __init__(self, prior: Prior):
        """
        A factor that wraps a prior such that is can be optimised
        by classic AutoFit optimisers.

        To do this it implements a prior_model and analysis.

        Parameters
        ----------
        prior
            A message/prior, usually of another factor. Prior factors
            are generated programmatically.
        """
        # TODO: Consider analytical solution rather than implementing optimisation
        super().__init__(prior.factor, x=prior, name=namer(self.__class__.__name__))
        self.prior = prior
        self.label = f"PriorFactor({prior.label})"

    @property
    def prior_model(self) -> CollectionPriorModel:
        """
        A trivial prior model to conform to the expected interface.
        """
        return CollectionPriorModel(self.prior)

    @property
    def analysis(self) -> "PriorFactor":
        """
        This is the analysis class for a PriorFactor
        """
        return self

    def log_likelihood_function(self, instance) -> float:
        """
        Compute the likelihood.

        The instance is a collection with a single argument expressing a
        possible value for this prior. The likelihood is computed by simply
        evaluating the prior's PDF for the given value.
        """
        return self.prior.factor(instance[0])

    @property
    def variable(self):
        return list(self.variables)[0]
