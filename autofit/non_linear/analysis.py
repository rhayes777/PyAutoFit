from abc import ABC

from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.result import Result
from autofit.non_linear.samples import OptimizerSamples


class Analysis(ABC):

    def log_likelihood_function(self, instance):
        raise NotImplementedError()

    def visualize(self, paths: AbstractPaths, instance, during_analysis):
        pass

    def save_attributes_for_aggregator(self, paths: AbstractPaths):
        pass

    def save_results_for_aggregator(self, paths: AbstractPaths, model: CollectionPriorModel,
                                    samples: OptimizerSamples):
        pass

    def make_result(self, samples, model, search):
        return Result(samples=samples, model=model, search=search)
