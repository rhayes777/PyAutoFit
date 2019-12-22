import math

import autofit as af
from autofit.optimize.non_linear.output import AbstractOutput


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis, model):
        mock_output = AbstractOutput(model, self.paths)
        mock_output.save_model_info()
        if model.prior_count == 0:
            raise AssertionError("There are no priors associated with the model!")
        if model.prior_count != len(model.unique_prior_paths):
            raise AssertionError(
                "Prior count doesn't match number of unique prior paths"
            )
        index = 0
        unit_vector = model.prior_count * [0.5]
        while True:
            try:
                instance = model.instance_from_unit_vector(unit_vector)
                fit = analysis.fit(instance)
                break
            except af.exc.FitException as e:
                unit_vector[index] += 0.1
                if unit_vector[index] >= 1:
                    raise e
                index = (index + 1) % model.prior_count
        return af.Result(
            instance,
            fit,
            model,
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(model.priors, key=lambda prior: prior.id)
            ],
        )


class MockAnalysis(af.Analysis):
    def fit(self, instance):
        return 1.0

    def visualize(self, instance, during_analysis):
        pass

    def __init__(self, data, results):
        self.data = data
        self.results = results
