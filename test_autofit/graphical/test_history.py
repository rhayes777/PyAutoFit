from copy import copy

import pytest

import autofit as af
from autofit import graphical as g, EPHistory
from autofit.graphical.declarative.result import EPResult
from autofit.graphical.expectation_propagation import FactorHistory


class Analysis(af.Analysis):

    def log_likelihood_function(self, instance):
        return -1.0


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Model(
        af.Gaussian,
        centre=af.GaussianPrior(mean=50, sigma=20),
        normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )


@pytest.fixture(
    name="factor_model"
)
def make_factor_model(model):
    return g.AnalysisFactor(model, analysis=Analysis())


@pytest.fixture(
    name="factor_history"
)
def make_history(factor_model):
    return FactorHistory(
        factor_model.model_factors[0]
    )


@pytest.fixture(
    name="result"
)
def make_result(model):
    # noinspection PyTypeChecker
    return af.Result(None, model=model)


@pytest.fixture(
    name="good_history"
)
def make_factor_history(factor_history, result):
    good_history = copy(factor_history)
    good_history(None, g.Status(result=result))
    return good_history


# noinspection PyTypeChecker
def test_factor_history(
        good_history,
        result
):
    assert good_history.latest_result is result


def test_bad_history(factor_history):
    # noinspection PyTypeChecker
    factor_history(None, g.Status(success=False))
    with pytest.raises(
            af.exc.HistoryException
    ):
        assert factor_history.latest_result


def test_latest_results(
        good_history,
        result,
        factor_model,
):
    history = EPHistory()
    history.history[factor_model] = good_history

    # noinspection PyTypeChecker
    ep_result = EPResult(
        ep_history=history,
        declarative_factor=factor_model,
        updated_ep_mean_field=None,
    )
    assert ep_result.latest_results == [result]
