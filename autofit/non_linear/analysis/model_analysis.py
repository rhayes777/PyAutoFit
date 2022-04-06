from .analysis import Analysis
from .combined import CombinedAnalysis


class ModelAnalysis(Analysis):
    def __init__(self, analysis, model):
        self.analysis = analysis
        self.model = model

    def __getattr__(self, item):
        return getattr(self.analysis, item)

    def log_likelihood_function(self, instance):
        return self.analysis.log_likelihood_function(instance)


class CombinedModelAnalysis(CombinedAnalysis):
    pass