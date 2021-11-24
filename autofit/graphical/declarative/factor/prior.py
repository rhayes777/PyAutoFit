from autofit.graphical.factor_graphs.factor import Factor
from autofit.mapper.prior.abstract import Prior
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.analysis import Analysis
from autofit.text.formatter import TextFormatter
from autofit.tools.namer import namer


class PriorFactor(Factor, Analysis):
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
        super().__init__(
            prior.factor,
            x=prior,
            name=namer(self.__class__.__name__)
        )
        self.prior = prior

    def make_results_text(self, model_approx):
        """
        Create a string describing the posterior values after this factor
        during or after an EPOptimisation.

        Parameters
        ----------
        model_approx: EPMeanField

        Returns
        -------
        A string containing the name of this factor and the current value of
        its single variable.
        """
        formatter = TextFormatter()
        formatter.add(
            (self.name,),
            model_approx.mean_field[
                self.prior
            ].mean
        )
        return formatter.text

    @property
    def prior_model(self) -> CollectionPriorModel:
        """
        A trivial prior model to conform to the expected interface.
        """
        return CollectionPriorModel(
            self.prior
        )

    @property
    def analysis(self) -> "PriorFactor":
        """
        This is the analysis class for a PriorFactor
        """
        return self

    def log_likelihood_function(
            self,
            instance
    ) -> float:
        """
        Compute the likelihood.

        The instance is a collection with a single argument expressing a
        possible value for this prior. The likelihood is computed by simply
        evaluating the prior's PDF for the given value.
        """
        return self.prior.factor(instance[0])
