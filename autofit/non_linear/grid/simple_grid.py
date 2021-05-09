from copy import copy

from autofit import Result
from autofit.mapper.prior_model import abstract
from autofit.mock.mock import MockSamples
from autofit.non_linear import abstract_search
from autofit.non_linear import paths
from autofit.non_linear.grid.grid_search import make_lists


class GridSearch:

    def __init__(self, step_size=0.5):

        self.step_size = step_size
        self.paths = paths.DirectoryPaths()

    def copy_with_paths(self, paths):
        search = copy(self)
        search.paths = paths
        return search

    def fit(
            self,
            model: abstract.AbstractPriorModel,
            analysis: abstract_search.Analysis
    ):
        best_likelihood = float("-inf")
        best_instance = None

        likelihoods = list()

        for list_ in make_lists(
                no_dimensions=model.prior_count,
                step_size=self.step_size
        ):
            instance = model.instance_from_unit_vector(
                list_
            )
            likelihood = analysis.log_likelihood_function(
                instance
            )
            likelihoods.append(likelihood)
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_instance = instance

        return Result(
            samples=MockSamples(
                max_log_likelihood_instance=best_instance,
                log_likelihood_list=likelihoods,
                gaussian_tuples=None
            ),
            model=model
        )
