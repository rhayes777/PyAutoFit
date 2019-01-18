from autofit.mapper.prior import UniformPrior
from autofit.optimize import non_linear
from autofit.optimize import optimizer


class GridSearch(non_linear.NonLinearOptimizer):
    def __init__(self, grid_priors, step_size=0.1, model_mapper=None, name="grid_search"):
        super().__init__(model_mapper=model_mapper, name=name)
        self.original_prior_dict = {prior: prior for prior in self.variable.priors}
        self.grid_priors = set(grid_priors)
        self.step_size = step_size

    @property
    def models_mappers(self):
        lists = optimizer.make_lists(len(self.grid_priors), step_size=self.step_size, include_upper_limit=False)
        for values in lists:
            priors = [UniformPrior(lower_limit=value, upper_limit=value + self.step_size) for value in values]
            arguments = {source_prior: prior for source_prior, prior in zip(self.grid_priors, priors)}
            yield self.variable.mapper_from_prior_arguments({**self.original_prior_dict, **arguments})
