from autofit.mapper.prior import UniformPrior
from autofit.optimize import optimizer


class GridSearch(object):
    def __init__(self, source_model_mapper, variables, step_size=0.1):
        self.source_model_mapper = source_model_mapper
        self.variables = variables
        self.step_size = step_size

    @property
    def models_mappers(self):
        lists = optimizer.make_lists(len(self.variables), step_size=self.step_size)
        for values in lists:
            priors = [UniformPrior(lower_limit=value, upper_limit=value + self.step_size) for value in values]
            arguments = {source_prior: prior for source_prior, prior in zip(self.variables, priors)}
            yield self.source_model_mapper.mapper_from_prior_arguments(arguments)
