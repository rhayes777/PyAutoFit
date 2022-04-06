from autofit.mapper.prior_model.collection import CollectionPriorModel
from .analysis import Analysis
from .free_parameter import IndexCollectionAnalysis


class ModelAnalysis(Analysis):
    def __init__(self, analysis, model):
        self.analysis = analysis
        self.model = model

    def __getattr__(self, item):
        return getattr(self.analysis, item)

    def log_likelihood_function(self, instance):
        return self.analysis.log_likelihood_function(instance)


class CombinedModelAnalysis(IndexCollectionAnalysis):
    def modify_model(self, model):
        return CollectionPriorModel([
            analysis.analysis.model if isinstance(
                analysis.analysis,
                ModelAnalysis
            ) else model
            for analysis in self.analyses
        ])
