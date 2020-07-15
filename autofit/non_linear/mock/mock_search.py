import math

from autoconf import conf
from autofit import exc
from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.paths import convert_paths
from autofit.non_linear.samples import PDFSamples


class MockSearch(NonLinearSearch):

    @convert_paths
    def __init__(self, paths=None, samples=None, fit_fast=True):

        super().__init__(paths=paths)

        self.fit_fast = fit_fast
        self.samples = samples or MockSamples()

    def _fit_fast(self, model, analysis):
        class Fitness:
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.log_likelihood_function(instance)
                self.result = MockResult(instance=instance)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(model.instance_from_vector)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    def _fit(self, model, analysis):
        if self.fit_fast:
            result = self._fit_fast(model=model, analysis=analysis)
            return result

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
            except exc.FitException as e:
                unit_vector[index] += 0.1
                if unit_vector[index] >= 1:
                    raise e
                index = (index + 1) % model.prior_count
        return MockResult(
            model=model,
            samples=MockSamples(
                log_likelihoods=fit,
                model=model,
                gaussian_tuples=[
                    (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                    for prior in sorted(model.priors, key=lambda prior: prior.id)
                ],
            ),
        )

    @property
    def config_type(self):
        return conf.instance.mock

    @property
    def tag(self):
        return "mock"

    def perform_update(self, model, analysis, during_analysis):
        return MockSamples(
            log_likelihoods=[1.0, 2.0],
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(model.priors, key=lambda prior: prior.id)
            ]
        )

    def samples_from_model(self, model):
        return MockSamples()

    @property
    def name(self):
        return "mock_search"


class MockAnalysis(Analysis):
    def log_likelihood_function(self, instance):
        return 1.0

    def visualize(self, instance, during_analysis):
        pass

    def __init__(self, data):
        self.data = data


class MockSamples(PDFSamples):
    def __init__(
            self,
            model=None,
            max_log_likelihood_instance=None,
            log_likelihoods=None,
            gaussian_tuples=None,
    ):

        if log_likelihoods is None:
            log_likelihoods = [1.0, 2.0, 3.0]

        super().__init__(
            model=model,
            parameters=[],
            log_likelihoods=log_likelihoods,
            log_priors=[],
            weights=[],
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self.gaussian_tuples = gaussian_tuples

    @property
    def max_log_likelihood_instance(self):
        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):
        return self.gaussian_tuples

    def write_table(self, filename):
        pass


class MockResult:
    def __init__(
            self,
            samples=None,
            instance=None,
            model=None,
            analysis=None,
            search=None,
    ):
        self.instance = instance or ModelInstance()
        self.model = model or ModelMapper()
        self.samples = samples or MockSamples(max_log_likelihood_instance=self.instance)

        self.previous_model = model
        self.gaussian_tuples = None
        self.analysis = analysis
        self.search = search

    def model_absolute(self, absolute):
        return self.model

    def model_relative(self, relative):
        return self.model

    @property
    def last(self):
        return self
