import pytest

import autofit as af


class Result(af.Result):
    pass


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 1.0 if isinstance(instance, af.Gaussian) else 0.0

    def make_result(self, samples, model, sigma=3.0, use_errors=True, use_widths=True):
        return Result(
            samples=samples,
            sigma=sigma,
            use_errors=use_errors,
            use_widths=use_widths,
        )


@pytest.fixture(name="Result")
def result_class():
    return Result


@pytest.fixture(name="Analysis")
def analysis_class():
    return Analysis


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.Gaussian)
