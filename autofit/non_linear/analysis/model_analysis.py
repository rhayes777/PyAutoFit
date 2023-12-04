from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior_model.collection import Collection
from .analysis import Analysis
from .indexed import IndexCollectionAnalysis


class ModelAnalysis(Analysis):
    def __init__(self, analysis: Analysis, model: AbstractPriorModel):
        """
        Comprises a model and an analysis that can be applied to instances of that model.

        Parameters
        ----------
        analysis
        model
        """
        self.analysis = analysis
        self.model = model

    def __getattr__(self, item):
        if item in ("__getstate__", "__setstate__"):
            raise AttributeError(item)
        return getattr(self.analysis, item)

    def log_likelihood_function(self, instance):
        return self.analysis.log_likelihood_function(instance)

    def make_result(self, samples):
        """
        Return the correct type of result by calling the underlying analysis.
        """
        return self.analysis.make_result(
            samples=samples,
        )


class CombinedModelAnalysis(IndexCollectionAnalysis):
    def modify_model(self, model: AbstractPriorModel) -> Collection:
        """
        Creates a collection with one model for each analysis. For each ModelAnalysis
        the model is used; for other analyses the default model is used.

        Parameters
        ----------
        model
            A default model

        Returns
        -------
        A collection of models, one for each analysis.
        """
        return Collection(
            [
                analysis.modify_model(analysis.analysis.model)
                if isinstance(analysis.analysis, ModelAnalysis)
                else analysis.modify_model(model)
                for analysis in self.analyses
            ]
        )
