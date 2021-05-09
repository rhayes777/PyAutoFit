import math
from typing import Optional

from autoconf import conf
from autofit import exc
from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.result import Result
from autofit.non_linear.samples import PDFSamples, Sample


class MockSearch(NonLinearSearch):
    def __init__(
            self,
            name="",
            unique_tag: Optional[str] = None,
            samples=None,
            fit_fast=True,
            sample_multiplier=1,
            **kwargs
    ):
        super().__init__(
            name=name,
            unique_tag=unique_tag,
            **kwargs
        )

        self.fit_fast = fit_fast
        self.samples = samples or MockSamples()
        self.sample_multiplier = sample_multiplier

    @property
    def config_dict_search(self):
        return {}

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

    def _fit(self, model: AbstractPriorModel, analysis, log_likelihood_cap=None):
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
        samples = MockSamples(
            samples=samples_with_log_likelihood_list(self.sample_multiplier * fit),
            model=model,
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(model.priors, key=lambda prior: prior.id)
            ],
        )

        self.paths.save_samples(samples)

        return MockResult(
            model=model,
            samples=samples,
        )

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mock"]

    def perform_update(self, model, analysis, during_analysis):
        return MockSamples(
            samples=samples_with_log_likelihood_list([1.0, 2.0]),
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(model.priors, key=lambda prior: prior.id)
            ]
        )

    def samples_from(self, model):
        return MockSamples()

    @property
    def name(self):
        return "mock_search"


class MockAnalysis(Analysis):
    def log_likelihood_function(self, instance):
        return 1.0

    def visualize(self, paths, instance, during_analysis):
        pass

    def __init__(self, data):
        self.data = data


def samples_with_log_likelihood_list(
        log_likelihood_list
):
    return [
        Sample(
            log_likelihood=log_likelihood,
            log_prior=0,
            weight=0
        )
        for log_likelihood
        in log_likelihood_list
    ]


class MockSamples(PDFSamples):
    def __init__(
            self,
            model=None,
            samples=None,
            max_log_likelihood_instance=None,
            gaussian_tuples=None
    ):

        self._samples = samples

        super().__init__(
            model=model,
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self.gaussian_tuples = gaussian_tuples

    @property
    def samples(self):

        if self._samples is None:
            return samples_with_log_likelihood_list(
                [1.0, 2.0, 3.0]
            )

        return self._samples

    @property
    def max_log_likelihood_instance(self):
        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):
        return self.gaussian_tuples

    def write_table(self, filename):
        pass


class MockResult(Result):
    def __init__(
            self,
            samples=None,
            instance=None,
            model=None,
            analysis=None,
            search=None
    ):
        super().__init__(samples, None, search)
        self._instance = instance or ModelInstance()
        self.model = model or ModelMapper()
        self.samples = samples or MockSamples(max_log_likelihood_instance=self.instance)

        self.model = model
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
