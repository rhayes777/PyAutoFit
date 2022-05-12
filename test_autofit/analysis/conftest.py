import pytest

import autofit as af


class Result(af.Result):
    pass


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return 1.0 if isinstance(
            instance,
            af.Gaussian
        ) else 0.0

    def make_result(self, samples, model, search):
        prior_passer = search.prior_passer
        return Result(
            samples=samples,
            model=model,
            sigma=prior_passer.sigma,
            use_errors=prior_passer.use_errors,
            use_widths=prior_passer.use_widths,
        )


@pytest.fixture(
    name="Result"
)
def result_class():
    return Result


@pytest.fixture(
    name="Analysis"
)
def analysis_class():
    return Analysis


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(af.Gaussian)
