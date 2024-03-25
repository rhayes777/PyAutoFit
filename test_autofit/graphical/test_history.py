from copy import copy

import pytest

import autofit as af
from autofit import graphical as g, EPHistory
from autofit.graphical.declarative.result import EPResult, HierarchicalResult
from autofit.graphical.expectation_propagation import FactorHistory


class Analysis(af.Analysis):
    def log_likelihood_function(self, instance):
        return -1.0


@pytest.fixture(name="model")
def make_model():
    return af.Model(
        af.Gaussian,
        centre=af.GaussianPrior(mean=50, sigma=20),
        normalization=af.GaussianPrior(mean=25, sigma=10),
        sigma=af.GaussianPrior(mean=10, sigma=10),
    )


@pytest.fixture(name="factor")
def make_factor(model):
    return g.AnalysisFactor(model, analysis=Analysis())


@pytest.fixture(name="factor_history")
def make_history(factor):
    return FactorHistory(factor.model_factors[0])


@pytest.fixture(name="result")
def make_result(model):
    # noinspection PyTypeChecker
    return af.mock.MockResult(None)


@pytest.fixture(name="good_history")
def make_factor_history(factor_history, result):
    good_history = copy(factor_history)
    good_history(None, g.Status(result=result))
    return good_history


# noinspection PyTypeChecker
def test_factor_history(good_history, result):
    assert good_history.latest_result is result


def test_bad_history(factor_history):
    # noinspection PyTypeChecker
    factor_history(None, g.Status(success=False))
    with pytest.raises(af.exc.HistoryException):
        assert factor_history.latest_result


def test_latest_results(
    good_history,
    result,
    factor,
):
    history = EPHistory()
    history.history[factor] = good_history

    # noinspection PyTypeChecker
    ep_result = EPResult(
        ep_history=history,
        declarative_factor=factor,
        updated_ep_mean_field=None,
    )
    assert ep_result.latest_results == [result]
    assert ep_result.latest_for(factor) == result


def test_hierarchical_results(good_history, result, hierarchical_factor):
    history = EPHistory()
    for factor in hierarchical_factor.factors:
        history.history[factor] = good_history

    factor = hierarchical_factor.factors[0]
    # noinspection PyTypeChecker
    ep_result = EPResult(
        ep_history=history,
        declarative_factor=factor,
        updated_ep_mean_field=None,
    )

    assert ep_result.latest_results == [result]
    assert ep_result.latest_for(factor) == result
    assert isinstance(ep_result.latest_for(hierarchical_factor), HierarchicalResult)


@pytest.fixture(name="hierarchical_factor")
def make_hierarchical_factor():
    hierarchical_factor = g.HierarchicalFactor(
        af.GaussianPrior,
        mean=af.GaussianPrior(mean=0.0, sigma=1.0),
        sigma=af.GaussianPrior(
            mean=1.0,
            sigma=0.01,
        ),
    )

    hierarchical_factor.add_drawn_variable(af.UniformPrior())
    hierarchical_factor.add_drawn_variable(af.UniformPrior())
    hierarchical_factor.add_drawn_variable(af.UniformPrior())
    return hierarchical_factor


def generate_samples(model):
    parameters = [
        [
            0.0,
            1.0,
            2.0,
        ],
        [
            0.0,
            1.0,
            2.0,
        ],
        [
            0.0,
            1.0,
            2.0,
        ],
        [
            21.0,
            22.0,
            23.0,
        ],
        [
            0.0,
            1.0,
            2.0,
        ],
    ]

    return af.Samples(
        model=model,
        sample_list=af.Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=[1.0, 2.0, 3.0, 10.0, 5.0],
            log_prior_list=[0.0, 0.0, 0.0, 0.0, 0.0],
            weight_list=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )


@pytest.fixture(name="results")
def make_results(hierarchical_factor):
    return [
        af.mock.MockResult(
            samples=generate_samples(factor.prior_model),
            model=factor.prior_model,
        )
        for factor in hierarchical_factor.factors
    ]


@pytest.fixture(name="hierarchical_result")
def make_hierarchical_result(results):
    return HierarchicalResult(results)


def test_combine_samples(hierarchical_result, results):
    result = results[0]
    assert len(hierarchical_result.samples) == 3 * len(result.samples)


def test_model(hierarchical_result):
    model = hierarchical_result.model
    assert isinstance(model, af.AbstractPriorModel)
    assert model.cls == af.GaussianPrior


def test_instance(hierarchical_result):
    instance = hierarchical_result.instance
    assert isinstance(instance, af.GaussianPrior)
