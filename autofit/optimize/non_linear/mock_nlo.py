import math

import autofit as af


class MockNLO(af.NonLinearOptimizer):
    def fit(self, analysis):
        self.save_model_info()
        if self.variable.prior_count == 0:
            raise AssertionError("There are no priors associated with the variable!")
        if self.variable.prior_count != len(self.variable.unique_prior_paths):
            raise AssertionError(
                "Prior count doesn't match number of unique prior paths"
            )
        index = 0
        unit_vector = self.variable.prior_count * [0.5]
        while True:
            try:
                instance = self.variable.instance_from_unit_vector(unit_vector)
                fit = analysis.fit(instance)
                break
            except af.exc.FitException as e:
                unit_vector[index] += 0.1
                if unit_vector[index] >= 1:
                    raise e
                index = (index + 1) % self.variable.prior_count
        return af.Result(
            instance,
            fit,
            self.variable,
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(self.variable.priors, key=lambda prior: prior.id)
            ],
        )