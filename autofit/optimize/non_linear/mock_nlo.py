import math

import autofit as af
from autofit import AbstractSamples
from autofit.optimize.non_linear.non_linear import NonLinearOptimizer
from autofit.optimize.non_linear.non_linear import Analysis


class MockNLO(NonLinearOptimizer):
    def _fit(self, model, fitness_function):
        raise NotImplementedError()

    def _full_fit(self, model, analysis):

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
                fit = analysis.log_likelihood_function(instance)
                break
            except af.exc.FitException as e:
                unit_vector[index] += 0.1
                if unit_vector[index] >= 1:
                    raise e
                index = (index + 1) % model.prior_count
        return af.Result(
            previous_model=model,
            samples=MockSamples(
                log_likelihoods=fit,
                model=model,
                gaussian_tuples=[
                    (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                    for prior in sorted(model.priors, key=lambda prior: prior.id)
                ],
            ),
        )

    def samples_from_model(self, model):
        return MockOutput()

    @property
    def name(self):
        return "mock_nlo"

class MockOutput(object):
    def __init__(self):
        pass


class MockAnalysis(Analysis):
    def log_likelihood_function(self, instance):
        return 1.0

    def visualize(self, instance, during_analysis):
        pass

    def __init__(self, data):
        self.data = data


class MockSamples(AbstractSamples):

    def __init__(self, model=None, max_log_likelihood_instance=None, log_likelihoods=None, gaussian_tuples=None):

        super().__init__(model=model, parameters=[], log_likelihoods=[], log_priors=[], weights=[])

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self.log_likelihoods = log_likelihoods
        self.gaussian_tuples = gaussian_tuples

    @property
    def max_log_likelihood_instance(self) -> int:
        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):
        return self.gaussian_tuples