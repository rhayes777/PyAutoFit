import copy

import pytest

import autofit as af
from autofit import Paths
from test_autofit import mock


@pytest.fixture(name="phase")
def make_phase():
    phase = af.AbstractPhase(phase_name="phase name")
    phase.model.one = af.PriorModel(mock.Galaxy, light=mock.EllipticalLP)
    return phase


@pytest.fixture(name="model_promise")
def make_model_promise(phase):
    return phase.result.model.one.redshift


@pytest.fixture(name="constant_promise")
def make_constant_promise(phase):
    return phase.result.constant.one.redshift


@pytest.fixture(name="profile_promise")
def make_profile_promise(phase):
    return phase.result.model.one.light


@pytest.fixture(name="collection")
def make_collection():
    collection = af.ResultsCollection()
    model = af.ModelMapper()
    model.one = af.PriorModel(mock.Galaxy, light=mock.EllipticalLP)
    constant = af.ModelInstance()
    constant.one = mock.Galaxy(light=mock.EllipticalLP())

    result = mock.Result(model=model, constant=constant)

    model = af.ModelMapper()
    constant = af.ModelInstance()

    model.hyper_galaxy = mock.HyperGalaxy
    constant.hyper_galaxy = mock.HyperGalaxy()

    hyper_result = mock.Result(model=model, constant=constant)

    result.hyper_result = hyper_result

    collection.add("phase name", result)

    return collection


@pytest.fixture(name="last_model")
def make_last_model():
    return af.last.model.one.redshift


@pytest.fixture(name="last_constant")
def make_last_constant():
    return af.last.constant.one.redshift


class TestLastPromises:
    def test_model(self, last_model):
        assert last_model.path == ("one", "redshift")
        assert last_model.is_constant is False

    def test_constant(self, last_constant):
        assert last_constant.path == ("one", "redshift")
        assert last_constant.is_constant is True

    def test_recover_model(self, collection, last_model):
        result = last_model.populate(collection)

        assert result is collection[0].model.one.redshift

    def test_recover_constant(self, collection, last_constant):
        result = last_constant.populate(collection)

        assert result is collection[0].constant.one.redshift

    def test_recover_last_model(self, collection, last_model):
        last_results = copy.deepcopy(collection.last)
        collection.add("last_phase", last_results)

        result = last_model.populate(collection)
        assert result is last_results.model.one.redshift
        assert result is not collection[0].model.one.redshift

    def test_embedded_results(self, collection):
        hyper_result = af.last.hyper_result

        model_promise = hyper_result.model
        constant_promise = hyper_result.constant

        model = model_promise.populate(collection)
        constant = constant_promise.populate(collection)

        assert isinstance(model.hyper_galaxy, af.PriorModel)
        assert model.hyper_galaxy.cls is mock.HyperGalaxy
        assert isinstance(constant.hyper_galaxy, mock.HyperGalaxy)

    def test_raises(self, collection):
        bad_promise = af.last.model.a.bad.path
        with pytest.raises(AttributeError):
            bad_promise.populate(collection)


class TestCase:
    def test_model_promise(self, model_promise, phase):
        assert isinstance(model_promise, af.Promise)
        assert model_promise.path == ("one", "redshift")
        assert model_promise.is_constant is False
        assert model_promise.phase is phase

    def test_constant_promise(self, constant_promise, phase):
        assert isinstance(constant_promise, af.Promise)
        assert constant_promise.path == ("one", "redshift")
        assert constant_promise.is_constant is True
        assert constant_promise.phase is phase

    def test_non_existent(self, phase):
        with pytest.raises(AttributeError):
            assert phase.result.model.one.bad

        with pytest.raises(AttributeError):
            assert phase.result.constant.one.bad

    def test_recover_model(self, collection, model_promise):
        result = model_promise.populate(collection)

        assert result is collection[0].model.one.redshift

    def test_recover_constant(self, collection, constant_promise):
        result = constant_promise.populate(collection)

        assert result is collection[0].constant.one.redshift

    def test_populate_prior_model_model(self, collection, model_promise):
        new_galaxy = af.PriorModel(mock.Galaxy, redshift=model_promise)

        result = new_galaxy.populate(collection)

        assert result.redshift is collection[0].model.one.redshift

    def test_populate_prior_model_constant(self, collection, constant_promise):
        new_galaxy = af.PriorModel(mock.Galaxy, redshift=constant_promise)

        result = new_galaxy.populate(collection)

        assert result.redshift is collection[0].constant.one.redshift

    def test_kwarg_promise(self, profile_promise, collection):
        galaxy = af.PriorModel(mock.Galaxy, light=profile_promise)
        populated = galaxy.populate(collection)

        assert isinstance(populated.light, af.PriorModel)

        instance = populated.instance_from_prior_medians()

        assert isinstance(instance.kwargs["light"], mock.EllipticalLP)

    def test_embedded_results(self, phase, collection):
        hyper_result = phase.result.hyper_result

        assert isinstance(hyper_result, af.PromiseResult)

        model_promise = hyper_result.model
        constant_promise = hyper_result.constant

        print(model_promise.path)

        assert isinstance(model_promise.hyper_galaxy, af.Promise)
        assert isinstance(constant_promise.hyper_galaxy, af.Promise)

        model = model_promise.populate(collection)
        constant = constant_promise.populate(collection)

        assert isinstance(model.hyper_galaxy, af.PriorModel)
        assert model.hyper_galaxy.cls is mock.HyperGalaxy
        assert isinstance(constant.hyper_galaxy, mock.HyperGalaxy)
