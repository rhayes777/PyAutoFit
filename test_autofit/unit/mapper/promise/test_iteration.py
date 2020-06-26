import pytest

import autofit as af
from test_autofit import mock


@pytest.fixture(name="prior_0")
def make_prior_0():
    return af.UniformPrior()


@pytest.fixture(name="prior_1")
def make_prior_1():
    return af.UniformPrior()


@pytest.fixture(name="model")
def make_model(prior_0, prior_1):
    model = af.Mapper()

    model.collection = af.Collection([prior_0, prior_1])

    return model


@pytest.fixture(name="phase")
def make_phase(model):
    return af.AbstractPhase(phase_name="phase name", model=model, search=af.MockSearch())


@pytest.fixture(name="results_collection")
def make_results_collection(model):
    collection = af.ResultsCollection()
    instance = af.Instance()
    instance.collection = [1, 2]

    result = af.MockResult(model=model, instance=instance)

    collection.add("phase name", result)

    return collection


class TestIteration:
    def test_index_type(self, phase):
        promise_0 = phase.result.model.collection[0]
        promise_1 = phase.result.model.collection[1]

        assert isinstance(promise_0, af.prior.Promise)
        assert isinstance(promise_1, af.prior.Promise)

    def test_index_populate_model(self, phase, prior_0, prior_1, results_collection):
        promise_0 = phase.result.model.collection[0]
        promise_1 = phase.result.model.collection[1]

        prior = promise_0.populate(results_collection)
        assert prior == prior_0

        prior = promise_1.populate(results_collection)
        assert prior == prior_1

    def test_index_populate_instance(self, phase, prior_0, prior_1, results_collection):
        promise_0 = phase.result.instance.collection[0]
        promise_1 = phase.result.instance.collection[1]

        value = promise_0.populate(results_collection)
        assert value == 1

        value = promise_1.populate(results_collection)
        assert value == 2

    def test_iteration(self, phase):
        promises = list(phase.result.model.collection)

        assert len(promises) == 2
        assert all([isinstance(promise, af.prior.Promise) for promise in promises])
