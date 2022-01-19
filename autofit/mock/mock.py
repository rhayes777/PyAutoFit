import math
import numpy as np
from typing import Optional

from autoconf import conf

from autofit.non_linear.analysis import Analysis
from autofit.mapper.model import ModelInstance
from autofit.mapper.model_mapper import ModelMapper
from autofit.non_linear.abstract_search import NonLinearSearch
from autofit.non_linear.result import Result
from autofit.non_linear.samples import PDFSamples, Sample, NestSamples

from autofit import exc


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


class MockAnalysis(Analysis):
    prior_count = 2

    def __init__(self):
        super().__init__()
        self.fit_instances = list()

    def log_likelihood_function(self, instance):
        self.fit_instances.append(instance)
        return [1]

    def visualize(self, paths, instance, during_analysis):
        pass

    def log(self, instance):
        pass


class MockResult(Result):
    def __init__(
            self,
            samples=None,
            instance=None,
            model=None,
            analysis=None,
            search=None,
    ):

        super().__init__(samples, model, search)

        self._instance = instance or ModelInstance()
        self.model = model or ModelMapper()
        self.samples = samples or MockSamples(max_log_likelihood_instance=self.instance)

        self.gaussian_tuples = None
        self.analysis = analysis
        self.search = search
        self.model = model

    def model_absolute(self, absolute):
        return self.model

    def model_relative(self, relative):
        return self.model

    @property
    def last(self):
        return self


class MockSamples(PDFSamples):
    def __init__(
            self,
            model=None,
            sample_list=None,
            max_log_likelihood_instance=None,
            log_likelihood_list=None,
            gaussian_tuples=None,
            unconverged_sample_size=10,
            **kwargs,
    ):

        self._log_likelihood_list = log_likelihood_list

        self.model = model

        sample_list = sample_list or self.default_sample_list

        super().__init__(
            model=model, sample_list=sample_list, unconverged_sample_size=unconverged_sample_size, **kwargs
        )

        self._max_log_likelihood_instance = max_log_likelihood_instance
        self._gaussian_tuples = gaussian_tuples

    @property
    def default_sample_list(self):

        if self._log_likelihood_list is not None:
            log_likelihood_list = self._log_likelihood_list
        else:
            log_likelihood_list = range(3)

        return [
            Sample(
                log_likelihood=log_likelihood,
                log_prior=0.0,
                weight=0.0
            )
            for log_likelihood
            in log_likelihood_list
        ]

    @property
    def log_likelihood_list(self):

        if self._log_likelihood_list is None:
            return super().log_likelihood_list

        return self._log_likelihood_list

    @property
    def max_log_likelihood_instance(self):

        if self._max_log_likelihood_instance is None:

            try:
                return super().max_log_likelihood_instance
            except (KeyError, AttributeError):
                pass

        return self._max_log_likelihood_instance

    def gaussian_priors_at_sigma(self, sigma=None):

        if self._gaussian_tuples is None:
            return super().gaussian_priors_at_sigma(sigma=sigma)

        return self._gaussian_tuples

    def write_table(self, filename):
        pass


class MockNestSamples(NestSamples):

    def __init__(
            self,
            model,
            sample_list=None,
            total_samples=10,
            log_evidence=0.0,
            number_live_points=5,
            time: Optional[float] = None,
    ):

        self.model = model

        if sample_list is None:

            sample_list = [
                Sample(
                    log_likelihood=log_likelihood,
                    log_prior=0.0,
                    weight=0.0
                )
                for log_likelihood
                in self.log_likelihood_list
            ]

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time
        )

        self._total_samples = total_samples
        self._log_evidence = log_evidence
        self._number_live_points = number_live_points

    @property
    def total_samples(self):
        return self._total_samples

    @property
    def log_evidence(self):
        return self._log_evidence

    @property
    def number_live_points(self):
        return self._number_live_points


class MockSearch(NonLinearSearch):
    def __init__(
            self,
            name="",
            samples=None,
            result=None,
            unique_tag: Optional[str] = None,
            prior_passer=None,
            fit_fast=True,
            sample_multiplier=1,
            save_for_aggregator=False,
            return_sensitivity_results=False,
            **kwargs
    ):

        super().__init__(name=name, unique_tag=unique_tag, **kwargs)

        self.samples = samples or MockSamples()
        self.result = result or MockResult(samples=samples)

        if prior_passer is not None:
            self.prior_passer = prior_passer

        self.fit_fast = fit_fast
        self.sample_multiplier = sample_multiplier

        self.save_for_aggregator = save_for_aggregator
        self.return_sensitivity_results = return_sensitivity_results

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mock"]

    @property
    def config_dict_search(self):
        return {}

    def _fit_fast(self, model, analysis, log_likelihood_cap=None):
        class Fitness:
            def __init__(self, instance_from_vector, result):
                self.result = result
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.log_likelihood_function(instance)

                if self.result.instance is None:
                    self.result.instance = instance

                # Return Chi squared
                return -2 * log_likelihood

        if self.save_for_aggregator:
            analysis.save_attributes_for_aggregator(paths=self.paths)

        fitness_function = Fitness(model.instance_from_vector, result=self.result)
        fitness_function(model.prior_count * [0.8])

        return fitness_function.result

    def _fit(self, model, analysis, log_likelihood_cap=None):

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
            sample_list=samples_with_log_likelihood_list(self.sample_multiplier * fit),
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

    def perform_update(self, model, analysis, during_analysis):

        if self.samples is not None and not self.return_sensitivity_results:
            self.paths.save_object("samples", self.samples)
            return self.samples

        return MockSamples(
            sample_list=samples_with_log_likelihood_list([1.0, 2.0]),
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(model.priors, key=lambda prior: prior.id)
            ]
        )
