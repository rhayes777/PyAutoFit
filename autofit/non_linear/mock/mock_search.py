import math
from typing import Optional, Tuple

from autoconf import conf
from autofit import exc
from autofit.mapper.model_mapper import ModelMapper
from autofit.graphical import FactorApproximation
from autofit.graphical.utils import Status
from autofit.non_linear.search.abstract_search import NonLinearSearch
from autofit.non_linear.mock.mock_result import MockResult
from autofit.non_linear.mock.mock_samples import MockSamples
from autofit.non_linear.samples import Sample


def samples_with_log_likelihood_list(log_likelihood_list, kwargs):
    if isinstance(log_likelihood_list, float):
        log_likelihood_list = [log_likelihood_list]
    return [
        Sample(
            log_likelihood=log_likelihood,
            log_prior=0,
            weight=0,
            kwargs=kwargs,
        )
        for log_likelihood in log_likelihood_list
    ]


def _make_samples(model):
    return {path: prior.value_for(0.5) for path, prior in model.path_priors_tuples}


class MockSearch(NonLinearSearch):
    def __init__(
        self,
        name="",
        samples=None,
        result=None,
        unique_tag: Optional[str] = None,
        fit_fast=True,
        sample_multiplier=1,
        save_for_aggregator=False,
        return_sensitivity_results=False,
        **kwargs
    ):
        super().__init__(name=name, unique_tag=unique_tag, **kwargs)

        self.samples = samples or MockSamples(ModelMapper())

        self.result = MockResult(samples=samples) if result is None else result

        self.fit_fast = fit_fast
        self.sample_multiplier = sample_multiplier

        self.save_for_aggregator = save_for_aggregator
        self.return_sensitivity_results = return_sensitivity_results

    def check_model(self, model):
        pass

    @property
    def config_type(self):
        return conf.instance["non_linear"]["mock"]

    @property
    def config_dict_search(self):
        return {}

    def _fit_fast(self, model, analysis):
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

        self.paths.samples_to_csv(self.samples)

        if self.save_for_aggregator:
            analysis.save_attributes(paths=self.paths)

        fitness = Fitness(model.instance_from_vector, result=self.result)
        fitness([prior.mean for prior in model.priors_ordered_by_id])

        return fitness.result

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
        samples = MockSamples(
            sample_list=samples_with_log_likelihood_list(
                self.sample_multiplier * fit, _make_samples(model)
            ),
            model=model,
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(model.priors, key=lambda prior: prior.id)
            ],
        )

        self.paths.samples_to_csv(self.samples)

        return analysis.make_result(
            samples=samples,
        )

    def perform_update(self, model, analysis, during_analysis, search_internal=None):
        if self.samples is not None and not self.return_sensitivity_results:
            self.paths.samples_to_csv(self.samples)
            return self.samples

        return MockSamples(
            sample_list=samples_with_log_likelihood_list(
                [1.0, 2.0], _make_samples(model)
            ),
            gaussian_tuples=[
                (prior.mean, prior.width if math.isfinite(prior.width) else 1.0)
                for prior in sorted(model.priors, key=lambda prior: prior.id)
            ],
            model=model,
        )

    def samples_from(self, model):
        return self.samples


class MockOptimizer(MockSearch):
    def __init__(self, **kwargs):
        super().__init__(fit_fast=False, **kwargs)

    @property
    def samples_cls(self):
        return MockOptimizer

    def project(
        self, factor_approx: FactorApproximation, status: Status = Status()
    ) -> Tuple[FactorApproximation, Status]:
        pass

    init_args = list()
